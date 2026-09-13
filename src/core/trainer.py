import os
import contextlib
import torch
import torch.nn as nn
from src.utils.metrics import ThroughputMeter, GPUUtilizationTracker

class Trainer:
    """
    Core training loop for Crimson.
    Handles forward/backward passes, metric tracking, logging, and checkpointing.
    Built to handle both single-GPU and DDP environments gracefully.
    """
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
        rank,
        logger,
        cfg,
        train_sampler=None,
        scheduler=None,
        start_epoch=0,
        best_acc=0.0
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.rank = rank
        self.logger = logger
        self.cfg = cfg
        self.train_sampler = train_sampler
        self.scheduler = scheduler
        
        self.epochs = cfg.model.epochs
        self.start_epoch = start_epoch
        self.output_dir = cfg.output_dir
        self.best_acc = best_acc
        
        self.accumulation_steps = getattr(cfg.hardware, "accumulation_steps", 1)
        
        self.throughput_meter = ThroughputMeter()
        self.gpu_tracker = GPUUtilizationTracker()
        
        # Automatic Mixed Precision (AMP) setup
        self.use_amp = getattr(cfg.hardware, "use_amp", False) and torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    def train(self):
        """Executes the full training and validation loop across all epochs."""
        for epoch in range(self.start_epoch, self.epochs):
            # Critical for DDP: forces the sampler to reshuffle data differently each epoch
            if self.train_sampler:
                self.train_sampler.set_epoch(epoch)
                
            train_loss, train_acc = self._train_epoch(epoch)
            val_loss, val_acc_top1, val_acc_top5 = self._validate_epoch(epoch)
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            if self.rank == 0:
                print(f"Epoch {epoch+1}/{self.epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Top-1: {val_acc_top1:.2f}% | Val Top-5: {val_acc_top5:.2f}%")
                
                self.logger.log_scalar("train/accuracy", train_acc, epoch)
                
                if self.scheduler is not None:
                    current_lr = self.scheduler.get_last_lr()[0]
                    self.logger.log_scalar("train/lr", current_lr, epoch)
                
                # Checkpointing logic: save only if validation accuracy improves
                if val_acc_top1 > self.best_acc:
                    self.best_acc = val_acc_top1
                    self._save_checkpoint(epoch, val_acc_top1, is_best=True)
                
                # Fault Tolerance: Always save the latest epoch to resume from preemption
                self._save_checkpoint(epoch, val_acc_top1, is_best=False)

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        self.throughput_meter.start()
        self.gpu_tracker.reset()
        
        self.optimizer.zero_grad()
        
        # PyTorch Profiler setup (only profile the first 5 steps of the first epoch if enabled)
        profiler = None
        if epoch == 0 and getattr(self.cfg.hardware, "profile", False):
            os.makedirs("results/profiler", exist_ok=True)
            
            def trace_handler(p):
                print(f"[Rank {self.rank}] Exporting Chrome trace to results/profiler/trace_rank{self.rank}.json...")
                p.export_chrome_trace(f"results/profiler/trace_rank{self.rank}.json")
                
            activities = [torch.profiler.ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(torch.profiler.ProfilerActivity.CUDA)
                
            profiler = torch.profiler.profile(
                activities=activities,
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
                on_trace_ready=trace_handler,
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            )
            profiler.start()
        
        for step, (images, labels) in enumerate(self.train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Use no_sync() to prevent All-Reduce overhead during gradient accumulation
            is_sync_step = (step + 1) % self.accumulation_steps == 0 or (step + 1) == len(self.train_loader)
            sync_context = self.model.no_sync() if not is_sync_step and hasattr(self.model, "no_sync") else contextlib.nullcontext()
            
            with sync_context:
                # Forward pass with Automatic Mixed Precision (AMP)
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    # Scale loss by accumulation steps to maintain effective learning rate
                    loss = loss / self.accumulation_steps
                
                # Backward pass with GradScaler
                self.scaler.scale(loss).backward()
            
            # Perform optimizer step only after accumulation_steps or at epoch end
            if (step + 1) % self.accumulation_steps == 0 or (step + 1) == len(self.train_loader):
                # Unscale gradients for clipping before stepping
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.model.max_grad_norm)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            
            # Metrics calculation
            total_loss += loss.item() * self.accumulation_steps
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Hardware tracking
            self.throughput_meter.update(images.size(0))
            # Defensive device_id: handles cuda:0 vs cuda (no explicit index) gracefully
            device_id = self.device.index if (self.device.type == "cuda" and self.device.index is not None) else (0 if self.device.type == "cuda" else None)
            gpu_util = self.gpu_tracker.sample(device_id)
            
            global_step = epoch * len(self.train_loader) + step
            
            # Log to TensorBoard every 50 steps (only on Rank 0)
            if self.rank == 0 and step % 50 == 0:
                self.logger.log_scalar("train/loss", loss.item() * self.accumulation_steps, global_step)
                self.logger.log_scalar("hardware/throughput", self.throughput_meter.get_throughput(), global_step)
                self.logger.log_gpu_utilization(gpu_util, global_step)
            
            if profiler is not None:
                profiler.step()
                if step >= 4:  # schedule=wait(1)+warmup(1)+active(3) = 5 total steps (0-4)
                    profiler.stop()
                    profiler = None
                
        return total_loss / len(self.train_loader), 100. * correct / total

    def _validate_epoch(self, epoch):
        self.model.eval()
        total_loss = 0.0
        correct_top1 = 0
        correct_top5 = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                
                # Top-1 accuracy
                _, predicted_top1 = outputs.max(1)
                correct_top1 += predicted_top1.eq(labels).sum().item()
                
                # Top-5 accuracy: check if true label is in top-5 predictions
                _, predicted_top5 = outputs.topk(5, dim=1, largest=True, sorted=True)
                correct_top5 += predicted_top5.eq(labels.view(-1, 1).expand_as(predicted_top5)).any(dim=1).sum().item()
                
                total += labels.size(0)
                
        # Aggregate validation metrics across all GPUs
        is_ddp = getattr(self.cfg.hardware, "mode", "single") == "ddp"
        if is_ddp and torch.distributed.is_initialized():
            import torch.distributed as dist
            metrics = torch.tensor([total_loss, correct_top1, correct_top5, total], dtype=torch.float32, device=self.device)
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            total_loss, correct_top1, correct_top5, total = metrics.tolist()
            correct_top1, correct_top5, total = int(round(correct_top1)), int(round(correct_top5)), int(round(total))
            num_batches = len(self.val_loader) * dist.get_world_size()
        else:
            num_batches = len(self.val_loader)

        val_loss = total_loss / num_batches
        val_acc_top1 = 100. * correct_top1 / total
        val_acc_top5 = 100. * correct_top5 / total
        
        # Log validation metrics
        if self.rank == 0:
            self.logger.log_scalar("val/loss", val_loss, epoch)
            self.logger.log_scalar("val/top1_accuracy", val_acc_top1, epoch)
            self.logger.log_scalar("val/top5_accuracy", val_acc_top5, epoch)
            
        return val_loss, val_acc_top1

    def _save_checkpoint(self, epoch, val_acc, is_best=True):
        os.makedirs(self.output_dir, exist_ok=True)
        filename = "best_model.pth" if is_best else "epoch_latest.pth"
        checkpoint_path = os.path.join(self.output_dir, filename)
        
        # Unwrap DDP model for clean saving (prevents 'module.' prefix issues on load)
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        
        torch.save({
            "epoch": epoch,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "val_acc": val_acc,
        }, checkpoint_path)
        
        if is_best:
            print(f"[Rank 0] Saved new best checkpoint to {checkpoint_path} (Acc: {val_acc:.2f}%)")

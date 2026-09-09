import os
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
        scheduler=None
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
        self.output_dir = cfg.output_dir
        self.best_acc = 0.0
        
        self.accumulation_steps = getattr(cfg.hardware, "accumulation_steps", 1)
        
        self.throughput_meter = ThroughputMeter()
        self.gpu_tracker = GPUUtilizationTracker()
        
        # Automatic Mixed Precision (AMP) setup
        self.use_amp = getattr(cfg.hardware, "use_amp", False) and torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    def train(self):
        """Executes the full training and validation loop across all epochs."""
        for epoch in range(self.epochs):
            # Critical for DDP: forces the sampler to reshuffle data differently each epoch
            if self.train_sampler:
                self.train_sampler.set_epoch(epoch)
                
            train_loss, train_acc = self._train_epoch(epoch)
            val_loss, val_acc = self._validate_epoch(epoch)
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            if self.rank == 0:
                print(f"Epoch {epoch+1}/{self.epochs} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.2f}%")
                
                if self.scheduler is not None:
                    current_lr = self.scheduler.get_last_lr()[0]
                    self.logger.log_scalar("train/lr", current_lr, epoch)
                
                # Checkpointing logic: save only if validation accuracy improves
                if val_acc > self.best_acc:
                    self.best_acc = val_acc
                    self._save_checkpoint(epoch, val_acc)

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
            gpu_util = self.gpu_tracker.sample(self.device.index if self.device.type == "cuda" else None)
            
            global_step = epoch * len(self.train_loader) + step
            
            # Log to TensorBoard every 50 steps (only on Rank 0)
            if self.rank == 0 and step % 50 == 0:
                self.logger.log_scalar("train/loss", loss.item() * self.accumulation_steps, global_step)
                self.logger.log_scalar("hardware/throughput", self.throughput_meter.get_throughput(), global_step)
                self.logger.log_gpu_utilization(gpu_util, global_step)
            
            # Step the profiler and stop after 5 steps
            if profiler is not None:
                profiler.step()
                if step >= 5:
                    profiler.stop()
                    profiler = None
                
        return total_loss / len(self.train_loader), 100. * correct / total

    def _validate_epoch(self, epoch):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        val_loss = total_loss / len(self.val_loader)
        val_acc = 100. * correct / total
        
        # Log validation metrics
        if self.rank == 0:
            self.logger.log_scalar("val/loss", val_loss, epoch)
            self.logger.log_scalar("val/accuracy", val_acc, epoch)
            
        return val_loss, val_acc

    def _save_checkpoint(self, epoch, val_acc):
        os.makedirs(self.output_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.output_dir, "best_model.pth")
        
        # Unwrap DDP model for clean saving (prevents 'module.' prefix issues on load)
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        
        torch.save({
            "epoch": epoch,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_acc": val_acc,
        }, checkpoint_path)
        
        print(f"[Rank 0] Saved new best checkpoint to {checkpoint_path} (Acc: {val_acc:.2f}%)")

# Training Logbook

The definitive record of training configurations, execution commands, and final benchmark results for the Crimson project.

---

## Production Configuration (100-Epoch Convergence)

| Hyperparameter / Flag | Default Value | Description |
| :--- | :--- | :--- |
| `model.name` | `resnet50` | Neural network architecture |
| `model.num_classes` | `100` | Target class count for CIFAR-100 |
| `model.learning_rate` | `0.1` | Initial learning rate for SGD (standard for ResNet trained from scratch) |
| `model.weight_decay` | `0.0001` | Weight decay coefficient |
| `model.epochs` | `100` | Number of training epochs (production convergence runs) |
| `model.warmup_epochs` | `5` | Linear LR warmup duration |
| `model.max_grad_norm` | `1.0` | Gradient clipping threshold |
| `dataset.batch_size` | `64` | Per-GPU batch size |
| `dataset.data_dir` | `./data` | Local directory for CIFAR-100 binary dataset |
| `hardware.mode` | `single` / `ddp` | Processing mode (`single` GPU vs `ddp` Distributed) |
| `hardware.backend` | `nccl` | PyTorch distributed backend |
| `hardware.use_amp` | `false` / `true` | Automatic Mixed Precision (FP16) toggle |
| `hardware.accumulation_steps` | `1` / `4` | Gradient accumulation micro-steps |
| `hardware.profile` | `false` / `true` | PyTorch Profiler Chrome trace toggle |

---

## 1. Final Convergence Benchmarks

These are the final, fully-converged metrics achieved on Kaggle's dual NVIDIA T4 environment using the production configuration listed above (including a modified 3x3 CIFAR stem and Cosine Annealing learning rate schedule).

| Hardware Config | Total Batch Size | Throughput (img/sec) | Final Top-1 Acc | Final Top-5 Acc | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1x NVIDIA T4 (Single-GPU, FP32, accum=1)** | 64 | 270 | **79.14%** | **94.60%** | Converged |
| **2x NVIDIA T4 (DDP, AMP FP16, accum=4)** | 512 | 869 | **78.42%** | **94.53%** | Converged |

**Key Performance Metrics:**
- **Speedup:** **3.22x** (DDP 2x T4 AMP FP16 vs 1x T4 FP32 baseline)
- **DDP Scaling Efficiency:** **93.5%** (Communication overhead $P_{comm} \approx 7.0\%$)
- **Accuracy Variance:** Minimal ~0.7% variance in Top-1 accuracy (79.14% vs 78.42%), which is mathematically expected due to the 8x larger effective global batch size (512 vs 64).

---

## 2. How to Run on Kaggle

**Step 1 - Enable GPU Accelerator**
Before running any cells, enable the GPU accelerator:
1. Click **Settings** in the top menu bar.
2. Hover over **Accelerator**.
3. Select **GPU T4 x2** from the submenu.

**Step 2 - Clone the Repo and Install Dependencies**
```python
!git clone https://github.com/sakshi-kadian/crimson.git
%cd crimson
!pip install -r requirements.txt
```

**Step 3 - Single-GPU Baseline**
```python
!torchrun --nproc_per_node=1 launcher.py hardware=single_gpu model.epochs=100
```

**Step 4 - Dual-GPU DDP Run (AMP + Gradient Accumulation + Profiler)**
```python
!torchrun --nproc_per_node=2 launcher.py hardware=ddp model.epochs=100 output_dir=./outputs_ddp
```

---

## 3. How to Save and Download Results from Kaggle

After all three cells finish running:

1. Click **"Save Version"** (top right of the Kaggle Notebook).
2. In the **"Version Type"** dropdown, select **"Quick Save"**.
3. Click **"Advanced Settings"**. In the **"Save Output"** dropdown, select **"Save output for this version when creating a Quick Save"**.
4. Click **Save**.
5. Once saved, go to the left sidebar and click **"Your Work"**.
6. Find your notebook and click the **"Output"** tab.
7. Click the **three dots (...)** next to the output folder.
8. Click **"Download output"** to download the entire output as a zip.
9. Extract the zip and copy these files and folders into your local repo:
   - `outputs/best_model.pth` (single-GPU best checkpoint, gitignored)
   - `outputs/epoch_latest.pth` (single-GPU latest checkpoint, gitignored)
   - `outputs_ddp/best_model.pth` (DDP best checkpoint, gitignored)
   - `outputs_ddp/epoch_latest.pth` (DDP latest checkpoint, gitignored)
   - `results/profiler/trace_rank0.json` (GPU 0 Chrome trace, gitignored)
   - `results/profiler/trace_rank1.json` (GPU 1 Chrome trace, gitignored)
   - `logs/tensorboard/` (local TensorBoard event files, gitignored)

---

## 4. Visualization and Analysis

### Viewing the Chrome Profiler Trace
To inspect the real GPU execution timeline, including microsecond-level NCCL communication and CUDA kernels:
1. Open **Google Chrome**.
2. Type `chrome://tracing` in the URL bar and press Enter.
3. Click **"Load"** and select `results/profiler/trace_rank0.json` or `trace_rank1.json`.

### Launching TensorBoard
To view training loss curves, throughput metrics, and GPU utilization:
```bash
tensorboard --logdir logs/tensorboard
```
Open your browser and navigate to: `http://localhost:6006`

---

## Final Notes

This logbook serves as the definitive record of Crimson's training history and reproduction steps. By following the exact configurations and commands documented above, any developer can replicate these benchmark results and independently verify the scaling performance of the DDP pipeline. 

For a deep dive into the technical reasoning behind these numbers, including the mathematical breakdown of communication bottlenecks and systems engineering lessons, please read the [Scaling Analysis and Engineering Retrospective](ENGINEERING_REPORT.md).

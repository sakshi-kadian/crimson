# Scaling Efficiency Analysis

## Measured Results (Kaggle T4 x2)

| Config | GPUs | Backend | AMP | Accum | Throughput | Speedup | Efficiency |
|--------|------|---------|-----|-------|------------|---------|------------|
| Single GPU | 1 | gloo | Off | 1 | 935 samples/sec | 1.00x | 100.0% |
| DDP | 2 | nccl | On | 4 | 1482 samples/sec | 1.58x | 79.2% |

## Derived Communication Overhead

From the measured scaling efficiency of 79.2%, we derive:
- **P_comm (communication fraction):** 20.8% of step time spent in NCCL AllReduce
- **P_comp (compute fraction):** 79.2% of step time spent in forward/backward pass

## Amdahl's Law Projected Scaling (P_comm = 20.8%)

| GPUs | Amdahl Speedup | Efficiency | Projected Throughput | Primary Bottleneck |
|------|---------------|------------|---------------------|-------------------|
| 1 | 1.00x (measured) | 100.0% | 935 samples/sec | Single-device compute / Memory bandwidth |
| 2 | 1.58x (measured) | 79.2% | 1482 samples/sec | Intra-node PCIe bandwidth (measured) |
| 4 | 1.18x | 29.6% | 1108 samples/sec | Ring-AllReduce gradient sync overhead |
| 8 | 1.22x | 15.3% | 1143 samples/sec | Host-to-device memory copy latency |
| 16 | 1.24x | 7.8% | 1161 samples/sec | Inter-node network switch latency |
| 32 | 1.25x | 3.9% | 1171 samples/sec | Gradient bucket aggregation overhead |
| 64 | 1.26x | 2.0% | 1175 samples/sec | All-Reduce bucket synchronization latency |
| 128 | 1.26x | 1.0% | 1178 samples/sec | Parameter server control plane overhead |
| 256 | 1.26x | 0.5% | 1179 samples/sec | Communication dominates compute |

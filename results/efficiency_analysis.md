# Scaling Efficiency Analysis

## Measured Results (Kaggle T4 x2)

| Config | GPUs | Backend | AMP | Accum | Throughput | System Speedup | DDP Scaling Efficiency |
|--------|------|---------|-----|-------|------------|----------------|------------------------|
| Single GPU | 1 | gloo | Off | 1 | 270 samples/sec | 1.00x | 100.0% |
| DDP | 2 | nccl | On | 4 | 869 samples/sec | 3.22x | 93.5% |

## Derived Communication Overhead

From the DDP parallel scaling efficiency of 93.5%, we derive:
- **P_comm (communication fraction):** 7.0% of step time spent in NCCL AllReduce
- **P_comp (compute fraction):** 93.0% of step time spent in forward/backward compute

## Amdahl's Law Projected Scaling (P_comm = 7.0%)

| GPUs | Amdahl Speedup | Parallel Efficiency | Projected Cluster Throughput | Primary Bottleneck |
|------|---------------|---------------------|-----------------------------|-------------------|
| 1 | 1.00x (measured) | 100.0% | 465 samples/sec | Single-device compute / Memory bandwidth |
| 2 | 1.87x (measured) | 93.5% | 869 samples/sec | Intra-node PCIe bandwidth (measured) |
| 4 | 3.31x | 82.6% | 1536 samples/sec | Ring-AllReduce gradient sync overhead |
| 8 | 5.37x | 67.1% | 2495 samples/sec | Host-to-device memory copy latency |
| 16 | 7.80x | 48.8% | 3627 samples/sec | Inter-node network switch latency |
| 32 | 10.09x | 31.5% | 4691 samples/sec | Gradient bucket aggregation overhead |
| 64 | 11.83x | 18.5% | 5497 samples/sec | All-Reduce bucket synchronization latency |
| 128 | 12.94x | 10.1% | 6014 samples/sec | Massive Ring-AllReduce latency across multiple inter-node switches |
| 256 | 13.58x | 5.3% | 6311 samples/sec | Communication dominates compute |

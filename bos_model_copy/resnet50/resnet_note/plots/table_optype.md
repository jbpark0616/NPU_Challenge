| op_type | n | time (ms) | % | cores | avg TFLOPS | avg GB/s (I/O) | avg GB/s (DRAM) | dominant bound |
|---|---:|---:|---:|---|---:|---:|---:|---|
| `conv2d` | 200 | 18.71 | 47.9% | [15, 16, 20] | 12.18 | 42.5 | 8.9 | compute |
| `matmul` | 340 | 11.47 | 29.4% | [14, 16] | 15.24 | 82.4 | 9.7 | compute |
| `halo` | 210 | 3.63 | 9.3% | [14, 16, 20] | — | 211.2 | 0.0 | null |
| `pool2_d` | 10 | 1.93 | 5.0% | [20] | 0.04 | 49.4 | 0.0 | compute |
| `binary` | 160 | 1.19 | 3.0% | [20] | 0.18 | 767.9 | 0.0 | compute |
| `interleaved_to_sharded` | 20 | 0.83 | 2.1% | [16] | — | 17.1 | 0.0 | compute |
| `reshard` | 210 | 0.52 | 1.3% | [14, 16, 20] | — | 944.1 | 0.0 | compute |
| `untilize_with_unpadding` | 30 | 0.29 | 0.7% | [16] | — | 44.8 | 0.0 | compute |
| `sharded_to_interleaved` | 20 | 0.17 | 0.4% | [16] | — | 66.1 | 0.0 | compute |
| `tilize` | 10 | 0.09 | 0.2% | [14] | — | 264.3 | 0.0 | compute |
| `tilize_with_val_padding` | 20 | 0.09 | 0.2% | [16] | — | 120.5 | 0.0 | compute |
| `reduce` | 10 | 0.08 | 0.2% | [16] | — | 102.4 | 0.0 | null |
| `move` | 20 | 0.06 | 0.2% | [16, 20] | — | 1791.9 | 0.0 | compute |

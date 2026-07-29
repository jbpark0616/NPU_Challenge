| quantity | value | 근거 |
|---|---:|---|
| Per-core peak (LoFi) at Eagle-N clock 650 MHz | 2.66 TOPS | 4096 mul-adds/cycle (matmul_device_operation.cpp:2173) × 650 MHz |
| Physical cores (Eagle-N, 6×4) | 24 | BOS spec (physical grid) |
| Compute-usable cores (Eagle-N, 5×4) | 20 | `project_eagle_n_hardware.md` (harvested 4 core 제외) |
| **Aggregate peak (BOS spec)** | **64 TOPS** | 2.66 × 24 — BOS 마케팅 '64 TOPS' 실체 |
| **Aggregate peak (effective)** | **53 TOPS** | 2.66 × 20 — 실제 baseline 가용치 |
| Peak DRAM BW (LPDDR5-6400 × 3 ch × 4) | ≈ 150 GB/s | 산술 계산치 (노트 §12.8) — 12-channel 구성이 Eagle-N 실제인지 BOS spec 확인 필요 |

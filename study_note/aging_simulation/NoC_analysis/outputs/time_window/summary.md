# ResNet-50 on Eagle-N A0 — 시간축 요약

- Total simulated (device-time): 4099.9 μs
- Bin size: 0.067 μs  (61102 bins)
- Clock: 650 MHz

## 1. NOC 0 : NOC 1 총 바이트 비 (real ops)

- **NOC 0: 3.9%**
- **NOC 1: 96.1%**
- 비대칭: **NOC 1 이 24.3× 지배**

## 2. 채널별 완전 유휴 시간

- NOC 0 rate == 0: **95.6%** of run
- NOC 1 rate == 0: 2.0% of run

## 3. 활성 compute 코어 분포 (of 20)

- mean 14.5, median 16, min 0, max 20

## 4. 관측 가능 창 (핵심 지표)

동시 조건: **여유 코어 ≥ N** AND **NOC 0 완전 유휴**

- 여유 코어 ≥ 4 AND NOC 0 == 0: **58.1%** of time
- 여유 코어 ≥ 8 AND NOC 0 == 0: **26.1%** of time

## 5. 최대 링크 rate 분위수 (real ops)

| NoC | p50 (B/μs) | p95 (B/μs) | max (B/μs) |
|---|---:|---:|---:|
| NOC 0 | 0.000 | 0.000 | 23043.815 |
| NOC 1 | 3699.977 | 12207.193 | 33922.447 |

**정합성 검증**: NOC0 max_link==0 pct **95.61%**, NOC0 total==0 pct **95.61%** → 일치

### 활성 bin 만 (rate > 0) 분위수

| NoC | active p50 | active p95 | active count |
|---|---:|---:|---:|
| NOC 0 | 2735.427 | 6233.286 | 2680 |
| NOC 1 | 3942.761 | 12207.193 | 59868 |

## 6. op 별 바이트 top 10 (real ops)

| # | op_type | duration μs | NOC0 bytes | NOC1 bytes | op_id |
|---|---|---:|---:|---:|---|
| 1 | Matmul | 136.94 | 0 | 4,259,840 | `op_stage4_rhid31744_Matmul` |
| 2 | Pool | 51.34 | 1,820,160 | 1,078,272 | `op_stage0_rhid05120_Pool2D` |
| 3 | Halo | 40.09 | 1,092,672 | 660,544 | `op_stage0_rhid04096_HaloDeviceOperation` |
| 4 | Conv2d | 101.72 | 10,240 | 1,164,160 | `op_stage4_rhid08192_Conv2d` |
| 5 | Halo | 10.32 | 605,280 | 255,168 | `op_stage2_rhid06144_HaloDeviceOperation` |
| 6 | Matmul | 63.19 | 0 | 774,656 | `op_stage4_rhid06144_Matmul` |
| 7 | Matmul | 61.57 | 0 | 774,656 | `op_stage4_rhid11264_Matmul` |
| 8 | Matmul | 63.29 | 0 | 774,656 | `op_stage4_rhid14336_Matmul` |
| 9 | Matmul | 61.27 | 0 | 774,656 | `op_stage4_rhid16384_Matmul` |
| 10 | Matmul | 63.37 | 0 | 774,656 | `op_stage4_rhid19456_Matmul` |

## 7. op 겹침 여부

- sequential (no overlap, verified from profile_log_device.csv)
- 각 stage 내 op 은 FastDispatch 순차 실행. Stage 간 host-gap 은 device-compressed timeline 에선 무시됨.

## 8. Warmup 제외 통계 (t > 300 μs)

초기 300 μs (stem broadcast + weight 로딩) 을 제외한 정상 실행 구간:

- 평균 활성 코어: **14.1** (median 16)
- NOC 0 완전 유휴: **96.4%**
- NOC 1 완전 유휴: 2.2%
- 관측 가능 창 (여유≥4 & NOC0 유휴): **62.7%**
- 관측 가능 창 (여유≥8 & NOC0 유휴): **28.1%**

## 좌표 신뢰도 주의

- 좌표는 logical (translated). 물리 배치와 1:1 대응 아님.
- 주 분석은 compute↔compute (real). y=3 sink 및 x=5 관련은 boundary 라벨로 분리.
- 절대 대역폭 (GB/s) 은 미확보. bytes/μs 원단위 유지.

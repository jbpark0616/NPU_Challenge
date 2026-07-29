# ResNet50 Stage-Split NoC 실측 → Aggregation → 시각화 전체 flow

Eagle-N (BOS P100 A0) 에서 ResNet50 을 5-stage 로 나눠 NoC 트랜잭션 실측하고,
코어별 / op별 / 시간축별 spatial 분석까지 뽑는 전체 파이프라인.

**2026-07-19 정리**. 지금까지 우리가 실제로 밟은 순서 그대로.

---

## 0. 전제

- SSH 서버 (`compasslab-...`) 에 `bos_host` 도커 컨테이너 있음
- 컨테이너 안에 tt-metal (`/tt-metal/`) + tt-npe (`/tt-npe/`) 설치돼 있음
- 로컬 (노트북) 에는 fork `bos_model_copy/resnet50_jbpark/` 가 있음
- 컨테이너 안에도 같은 fork 가 `/tt-metal/models/bos_model/resnet50_jbpark/` 에 있음
  - **주의**: 로컬 fork ↔ 컨테이너 fork 는 **별개 사본**. mount 아님. 수정하면 양쪽 다 반영 필요 (`docker cp` 로)
  - 관련 memory: `project-fork-container-sync`

---

## 1. 데이터 수집 — 컨테이너에서 stage 0~4 실행

### 1-1. 세션 진입 & 환경

```bash
docker exec -it bos_host /bin/bash

export PATH="$HOME/.local/bin:$PATH"
cd /tt-metal && source env_set.sh
source /tt-npe/ENV_SETUP

# sanity check
which tt_npe.py
which npe_analyze_noc_trace_dir.py
python -c "import ttnn; print('OK')"
```

### 1-2. stage 별 실행 (하나씩 확인하며)

Stage 0 (stem) 는 예전 데이터 재사용 가능. Stage 1~4 는 각각:

```bash
bash /tt-metal/models/bos_model/resnet50_jbpark/run_stage_split.sh 1
bash /tt-metal/models/bos_model/resnet50_jbpark/run_stage_split.sh 2
bash /tt-metal/models/bos_model/resnet50_jbpark/run_stage_split.sh 3
bash /tt-metal/models/bos_model/resnet50_jbpark/run_stage_split.sh 4
```

각 실행마다 새 timestamp 폴더 생김:
```
/tt-metal/generated/profiler/stage_split_<YYYYMMDD_HHMMSS>/stage_<N>/logs/
```

**성공 표지**:
- pytest `PASSED`
- `noc_trace_ID*_merged.json` 이 op 수만큼 생성됨
- tt-npe post-processing 크래시는 **예상됨. 무시** (multicast assertion + API skew)

### 1-3. 알려진 실행 오류 & 대처

| 오류 | 원인 | 해결 |
|--|--|--|
| `Tile layout is required for tensor of type bfloat8_b; got Layout::ROW_MAJOR` | `_synth_layer_input` 에서 bfloat8 + ROW_MAJOR 조합 불가 | `from_torch(dtype=ttnn.bfloat16, ...)` 로 수정. TILE 변환은 뒤의 `to_layout` 에서 |
| `Physical shard shape (56, 512) must be tile {32, 32} sized` | stage 3/4 spatial size 가 CORE_SET_14 로 tile-align 안 됨 | 14-core shard 못 만들면 DRAM interleaved 로 폴백 |
| `'NoneType' object has no attribute 'grid'` at `_forward_layer` shard cache | DRAM interleaved 입력 → module1 output 도 non-sharded | `shard_spec is not None` guard 추가 |

fix 는 이미 `tt/ttnn_functional_resnet50.py` 에 적용됨. 컨테이너 동기화만 잊지 말 것.

### 1-4. 컨테이너 → 서버 → 로컬 파일 전송

컨테이너에서 나가서 (서버 셸에서):

```bash
mkdir -p ~/npe_stages
for TS in <새 timestamp>/stage_1 <새 timestamp>/stage_2 ... ; do
    mkdir -p ~/npe_stages/$TS
    docker cp bos_host:/tt-metal/generated/profiler/stage_split_$TS ~/npe_stages/
done
```

서버 → 로컬은 사용자 편의 (VS Code SSH / scp / rsync). 최종 로컬 경로:
```
bos_model_copy/resnet50_jbpark/stage_split_<ts>/stage_<N>/logs/
```

### 1-5. 각 stage 폴더 필수 파일

```
stage_<N>/logs/
├── noc_trace_dev0_ID*.json     # raw NoC trace (per op, per device)
├── noc_trace_ID*_merged.json   # 병합본 (aggregation 입력)
├── tracy_ops_data.csv          # op 이름 매핑
├── tracy_ops_times.csv         # host-side op timing
├── profile_log_device.csv      # 코어×RISC 별 zone timing
├── topology.json               # (거의 비어있음, 우리는 안 씀)
└── cluster_coordinates.json    # (비어있음)
```

---

## 2. 로컬 aggregation — 코어/op별 통합 분석

### 2-1. 5개 stage 경로 매핑 갱신

로컬에서 새 stage 뽑을 때마다 `analysis/stage_paths.py` 의 매핑 timestamp 수정:

```python
STAGE_PATHS: dict[int, Path] = {
    0: _REPO_ROOT / "stage_split_20260712_173806" / "stage_0",
    1: _REPO_ROOT / "stage_split_20260718_143633" / "stage_1",
    ...
}
```

경로 검증:
```bash
cd bos_model_copy/resnet50_jbpark/analysis/
python stage_paths.py
# → [OK] all 5 stage paths valid + 각 폴더 op 수 출력
```

### 2-2. Aggregation 실행

```bash
python aggregate_per_core.py
```

출력 (`outputs/` 폴더):

| 파일 | 내용 |
|--|--|
| `per_core.csv` | 코어 좌표별 injected/consumed bytes, NoC0/1 별. **real ops만** |
| `per_core_all.csv` | 위 + boundary 포함 (비교용) |
| `per_op.csv` | op별 총 traffic, boundary 라벨 유지 |
| `matrix.csv` | (op, 코어) 셀별 상세 (long format) |
| `per_core_heatmap.png` | 6×5 grid 시각화 (real ops) |
| `per_core_heatmap_all.png` | 위 + boundary 포함 |
| `summary.txt` | 콘솔 요약 저장본 |

### 2-3. 시간축 slice heatmap

```bash
python time_slice_heatmaps.py
```

출력 (`outputs/` 안):

| 파일 | 내용 |
|--|--|
| `per_stage_heatmaps.png` | stage 0~4 각각 5행 × 2열 heatmap (shared vmax) |
| `cumulative_progression.png` | stem → +layer1 → ... 누적 |
| `per_op_top20_heatmaps.png` | top 20 real ops 각각의 injected spatial pattern |

### 2-4. Boundary artifact 필터 규칙

`boundary_filter.py` 에 이름 리스트:
- `InterleavedToShardedDeviceOperation`
- `ShardedToInterleavedDeviceOperation`
- `TilizeWithValPadding`, `Tilize`
- `UntilizeWithUnpadding`, `Untilize`
- `ReshardDeviceOperation`
- `MoveDeviceOperation`

Real op 통계 계산 시 제외. 총 boundary overhead 는 우리 데이터에서 **bytes 18.7%, cycles 20.6%**.

관련 memory: `project-stage-split-boundary-artifact`

---

## 3. tt-npe 시뮬레이터 분석 (별도 트랙)

우리 aggregation 은 **endpoint 실측** 만. Link-level congestion 분석은 tt-npe simulator 가 담당.

### 3-1. 컨테이너에서 5-stage 시뮬

```bash
# 세션 진입 후
for TS in 20260712_173806/stage_0 20260718_143633/stage_1 ... ; do
    LOGS=/tt-metal/generated/profiler/stage_split_$TS/logs
    OUT=/tt-metal/generated/profiler/stage_split_$TS/npe_summary.txt

    npe_analyze_noc_trace_dir.py "$LOGS" -e -m 200 -s -j 8 > "$OUT" 2>&1
done
```

옵션:
- `-e`: `.npeviz` timeline 파일 생성 → ttnn-visualizer 웹 (`ttnn-visualizer.tenstorrent.com/npe`) 에 드래그하면 링크별 utilization 시각화
- `-m 200`: 표 200행
- `-s`: cycle prediction accuracy 통계
- `-j 8`: 병렬 워커

### 3-2. 결과 로컬 전송

```bash
docker cp bos_host:/tt-metal/generated/profiler/stage_split_$TS/npe_summary.txt <서버 경로>
docker cp bos_host:/tt-metal/generated/profiler/stage_split_$TS/npe_viz <서버 경로>
# 서버 → 로컬은 사용자 편의
```

로컬 최종 위치:
```
bos_model_copy/resnet50_jbpark/stage_split_<ts>/stage_<N>/
├── npe_summary.txt
└── npe_viz/
    ├── manifest.json
    └── <opname>_ID<N>.npeviz
```

### 3-3. 요약 4줄 확인

`npe_summary.txt` 하단:
```
average link util                : X.X%
cycle-weighted overall link util : X.X%    ← 헤드라인
cycle-weighted dram bw util      : X.X%
cycle-weighted eth bw util       : X.X%    ← Eagle-N 은 ETH 없음, 아티팩트
```

Op 별 표에서 **`Cong Impact` > 5% 인 op** = 링크 실제 경합 존재.

---

## 4. 신뢰도 매트릭스 — 뭐 믿고 뭐 안 믿을지

### ✅ 100% 실측 (Eagle-N 하드웨어)

- Raw JSON 의 `sx/sy/dx/dy/num_bytes/timestamp/noc/type`
- 우리 aggregation 의 모든 수치 (`per_core.csv`, `per_op.csv`, `matrix.csv`)
- 코어별 injected/consumed bytes
- Op 별 cycles (하드웨어 사이클 카운터)

### ⚠️ tt-npe 시뮬 결과 (Blackhole reference 기반, 절대값 caveat 필수)

| 지표 | 절대값 | 상대 순위 |
|--|:--:|:--:|
| NoC Util % | ❌ | ✅ |
| DRAM BW Util % | ❌ **Blackhole 24 DRAM ↔ Eagle-N 3 LPDDR5. 정정 factor ~4.3배** | ✅ |
| Cong Impact % | ⚠️ topology mismatch | ✅ |
| Cycle prediction | ⚠️ 27~51% error | 🟡 |
| **ETH BW Util %** | ❌ **P100 은 ETH disabled. 완전 무시** | ❌ |

### ❌ 절대 인용 금지

- ETH BW Util (Eagle-N 에 ETH 없음)
- Blackhole 좌표 매핑을 우리 실측 좌표에 직접 대응
- tt-npe cycle prediction 개별 op 값

### 발표·논문 표현 원칙

- 실측 수치는 자신 있게 인용
- tt-npe % 는 "Blackhole reference peak 기준" 이라 명시
- DRAM util 은 factor 4.3 정정하거나 caveat
- 상대 순위·패턴은 신뢰

관련 memory: `project-eagle-n-noc-coords`, `roofline-debt`

---

## 5. 지금까지 확립한 사실 (2026-07-19)

1. **20 usable compute cores 실측 확인** — (0..4) × (1..4) 좌표 사각 배열
2. **DRAM 여유 매우 큼** — Blackhole 보정 후 ~5% util. **95% 여유**
3. **Layer3/4 시간대 NoC 완전 idle** — Cong Impact 0%
4. **Stem/layer2 short burst 에서만 real congestion** — Halo 21%, layer2 op 6~7%
5. **SubDevice free-riding 실측 근거** — wall time <1% 오버헤드 예상

## 6. 다음 실증 스텝

- [ ] Verifier microkernel 프로토타입 (SubDevice 4~6 코어)
- [ ] Verifier ON/OFF wall time 실측 → <1% 오버헤드 확인
- [ ] Power 실측 (있으면)
- [ ] tt-npe HAL translation dive 로 정확한 좌표 매핑 (task #9)

---

## 관련 문서

- `../CHANGES.md` — fork 변경 이력
- `../../study_note/stage_split_profiling_workflow.md` — 이전 compute-pass / NoC-pass 분할 문서
- `../../tutorials/guide/tt-npe_intro.md` — tt-npe 기초
- `../../tt-npe/tt-npe_install.md` — tt-npe 설치 노트 + 오늘 실행 로그
- Project memories: `project-fork-container-sync`, `project-stage-split-boundary-artifact`, `project-eagle-n-noc-coords`

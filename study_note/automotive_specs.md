# 차량용 인식 파이프라인 스펙 리서치

> 목적: NPU Challenge 12 core 구성이 낸 FPS/latency가 자율주행/ADAS의 요구치를 만족하는지 검증할 정량 기준 확보.
> 관련: `../README.md`, memory `project-automotive-validation`.

날짜: 2026-07-12 시작

---

## TL;DR

- **카메라 FPS의 실질 하한선은 ~30 FPS**. 프로덕션 시스템(Tesla HW4 24 FPS 카메라, DRIVE Orin 30 FPS 스테레오, 학술 관측 30~60 Hz 상한) 모두 이 근처.
- **End-to-end latency 예산은 ~100 ms가 스포츠형 기준, 350 ms까지는 실제 프로덕션 관측치**. Tesla는 photon→actuator 전 구간 <100 ms 목표 아래에서 인식(NN) 자체는 20~30 ms만 할당.
- **인식(perception) 단독 latency는 30~50 ms**가 사실상 상한. NVIDIA DLA/BI3D 30 ms, Tesla NN 20~30 ms, 학술 연구 <50 ms 목표.
- **함의**: 12 core 구성으로 ResNet50/YOLOv8s를 돌렸을 때 **≥30 FPS 유지 + 인식 latency ≤30 ms** 이면 프로덕션 급 요구치를 만족한다고 주장 가능. 이보다 낮으면 스펙 미달 가능성 있음 (재확인 필요).

⚠ 이 노트의 수치는 벤더 발표·기술 블로그·학술 논문 혼합 출처. 발표 인용 시 각 수치별 원 출처 재확인 요망.

---

## 비교표

| 시스템 | 카메라 개수 | 카메라 FPS | 입력 해상도 | E2E latency (photon→act) | 인식 NN latency | 컴퓨트 (TOPS) | Safety cert | 출처 |
|---|---|---|---|---|---|---|---|---|
| Tesla FSD HW3 | 8 | 36 | 1.2 MP | <100 ms 목표 | 20~30 ms (추정) | ~144 | 미공개 | [1][2] |
| Tesla FSD HW4 | 8 | 24 | ~5 MP | <100 ms | ~20~30 ms (v14.3 -20%) | 500+ | 미공개 | [1][2][3] |
| NVIDIA DRIVE Orin | 8 (per pipeline 예) | 30 (stereo perception KPI) | 다양 | — | 30 ms (BI3D DLA 예) | 254 | **ISO 26262 ASIL-D** | [4][5] |
| NVIDIA Jetson AGX Orin (DLA) | — | 46 (BI3D) | — | — | 30 ms | 254 | (모듈) | [4] |
| Mobileye EyeQ5H | 다중 카메라 | (미공개, 합산 처리) | 최대 8 MP | — | — | 24 (개별) | ASIL-B | [6][7] |
| Mobileye EyeQ6H | 다중 카메라 | 1000+ FPS **총합** (surround 통틀어) | 최대 8 MP | — | — | 34 (EyeQ5H의 3배) | ASIL-B/D | [6][7][8] |
| 프로덕션 dual-SoC AD-ECU (Saito et al.) | — | — | — | **296 ms 측정 / 350 ms 예산** | — | — | — | [9] |
| 학술 목표 (RT-BEV 등) | — | 30~60 Hz | — | 50~200 ms | <50 ms | — | — | [10] |

<br>

## 시스템별 상세

### Tesla FSD (HW3 / HW4)

**하드웨어**
- HW3: 2× Full Self-Driving 컴퓨터, 커스텀 NPU 12 core × 2 SoC, ~144 TOPS 총합.
- HW4: 2× SoC, RAM 16GB, storage 256GB, **약 500+ TOPS** (HW3 대비 3~5배 처리량).

**카메라 파이프라인**
- 8-camera surround.
- 프레임 레이트: **HW3 36 FPS → HW4 24 FPS로 감소**. 대신 해상도가 1.2 MP → ~5 MP로 상승, RGGB 컬러 센서 추가.
- 처리 throughput: HW3 = 1 gigapixel/s, HW4 = **1.3 gigapixel/s 추론**.

**Latency**
- Photon → actuator **전체 예산 <100 ms** (업계 관례에 부합).
- 세부 예산(비공식 breakdown):
  - 이미지 캡처: 10 ms
  - **NN 인식: 20~30 ms**
  - 인식 후처리: 10 ms
  - 경로 계획: 10 ms
  - 제어 출력: 10 ms
  - 액추에이터: 20~30 ms
- FSD v14.3에서 MLIR 컴파일러 스택 도입으로 **인식 latency 20% 감소** — 즉 HW4 기준 인식 20 ms 초반대로 진입.
- 최신 발표: HW4 상 sub-10 ms 추론까지 최적화된 경우도 언급됨 (AI5 대비 KPI).

<br>

### NVIDIA DRIVE Orin

**하드웨어**
- SoC: Arm CPU + Ampere GPU + 2× DLA(Deep Learning Accelerator).
- **254 TOPS INT8**.
- **ISO 26262 ASIL-D** 인증 (자율주행 최고 등급). NVIDIA 자체 발표.

**카메라 파이프라인**
- 다수 카메라(예: 8) 지원, DRIVE OS의 perception stack이 8-camera surround 입력 전제.
- Perception KPI 예: **스테레오 카메라 입력 30 FPS real-time detection**.
- Perception stack이 전체 컴퓨트의 60%+ 차지 (DNN heavy).

**Latency 예시**
- **BI3D (스테레오 disparity DNN 3개)를 DLA로 46 FPS + 30 ms latency** — Jetson AGX Orin 기준 (Orin SoC와 동일 실리콘).
- Perception 파이프라인 전체 latency는 "임계치 이하로 유지" 요구, 구체 수치는 whitepaper별 상이.

**BEV 인식 사례**
- NVAutoNet(BEV perception model)이 **DRIVE Orin에서 53 FPS** 실측.

<br>

### Mobileye EyeQ 시리즈

**EyeQ5**
- 24 TOPS급, ASIL-B, EyeQ5H가 상위 SKU.

**EyeQ6L (Lite)**
- 8 MP × 120° 카메라 지원.
- EyeQ4M 대비 4.5× 컴퓨트, 비슷한 전력.
- 프론트 카메라 기반 ADAS 진입점.

**EyeQ6H (High)**
- **EyeQ5H 대비 3× 컴퓨트, 25% 전력 증가에 그침** — 효율 크게 개선.
- 서라운드 카메라를 통틀어 **1000+ FPS 총 처리**(chip-wide aggregate).
- 자체 ISP + GPU + 비디오 인코더 통합.
- 하이레벨 자율주행(L2+/L3) 대상.

**EyeQ Ultra** (2025 로드맵)
- 진짜 L4/L5 대상 상위 라인. 이번 스코프에서는 상세 생략.

<br>

### 학술·프로덕션 latency 벤치마크

**Saito et al. (arxiv 2606.10856)**
- 실제 프로덕션 dual-SoC AD-ECU 상에서 **camera-shutter → planner-output 평균 296 ms, 350 ms 예산**.
- 이는 sensor exposure부터 계획 출력까지의 전 구간 → 인식만 떼면 이 중 절반 이하일 것.

**Fixstars ADAS 벤치마크 (일반)**
- **Perception-to-action 100~300 ms** 창(window)이 real-time ADAS 표준.
- "slack이 매우 작다" — 매 ms가 안전에 직결.

**RT-BEV / 최근 학술**
- 프로덕션은 50~200 ms end-to-end perception, 연구 목표는 **<50 ms**.
- **120 km/h에서 10 ms latency = 0.33 m 이동거리**.
- 카메라 30~60 Hz, LiDAR 10~20 Hz, radar 10~30 Hz.

<br>

## 자율주행 latency budget 개념

**Sensor-to-actuator latency** = 다음 합:
1. 센서 노출·판독 (~10 ms)
2. 이미지 신호 처리 (ISP, ~5~10 ms)
3. **Neural network 인식** (**20~50 ms — 우리 프로젝트가 영향 미치는 유일 구간**)
4. 인식 후처리 (NMS, tracking; ~10 ms)
5. 경로 계획 / 예측 (~10~30 ms)
6. 제어 명령 계산 (~10 ms)
7. 액추에이터 응답 (~20~30 ms)

전체 예산이 100 ms일 때 NN 인식에 **20~30 ms만 허용**되는 게 표준. NPU 성능 관점에서 이 "20~30 ms 인식 예산"이 핵심 목표치.

<br>

## 우리 baseline 대조 (템플릿, 실측 후 채움)

| 항목 | 요구치 (최소) | 요구치 (권장) | 우리 12 core 결과 | 여유 |
|---|---|---|---|---|
| **Perception FPS** | ≥ 24 (Tesla HW4 카메라 rate) | ≥ 30 (DRIVE Orin KPI) | (측정 후) | |
| **Perception latency** | ≤ 50 ms (학술 target) | ≤ 30 ms (Tesla NN 예산) | (측정 후) | |
| **Compute density (TOPS)** | (참고: EyeQ5H 24, EyeQ6H 34, Orin 254) | | Eagle-N 12-core 값 산출 필요 | |

**측정 계획**:
1. ResNet50 batch=1을 12 core로 재구성 후 iteration당 latency 실측.
2. YOLOv8s 동일.
3. Baseline(14~16 core) 대비 slowdown % 계산.
4. 위 표 채우고 "요구치 만족" / "요구치 미달 (부족 X ms)" 판정.

<br>

## Fault-tolerance 관점 (참고)

- ISO 26262 ASIL-D는 DRIVE Orin이 강조하는 지점 — TT 스택에서 subdevice로 유사 개념을 구현해야 함.
- 다음 리서치 항목(NVIDIA DRIVE Orin whitepaper의 FT 아키텍처)에서 상세 조사 예정.

<br>

## 출처

[1] [Tesla Autopilot hardware - Wikipedia](https://en.wikipedia.org/wiki/Tesla_Autopilot_hardware)
[2] [Tesla FSD Hardware 3 vs Hardware 4: Technical Deep Dive (LinkedIn)](https://www.linkedin.com/pulse/tesla-fsd-hardware-3-vs-4-technical-deep-dive-vikas-pandit-k23nc)
[3] [Tesla FSD v14.3 rolls out with MLIR rewrite, 20% faster reactions (Electrek)](https://electrek.co/2026/04/07/tesla-fsd-14-3-rolling-out-mlir-lattner/)
[4] [Maximizing Deep Learning Performance on NVIDIA Jetson Orin with DLA (NVIDIA)](https://developer.nvidia.com/blog/maximizing-deep-learning-performance-on-nvidia-jetson-orin-with-dla/)
[5] [NVIDIA Drive Orin: ADAS Use Cases (eInfochips)](https://www.einfochips.com/resources/publications/adas-and-autonomous-driving-use-cases-on-nvidia-drive-orin/)
[6] [The Evolution of EyeQ (Mobileye)](https://www.mobileye.com/technology/eyeq-chip/)
[7] [Meet EyeQ6: Our Most Advanced Driver-Assistance Chips Yet (Mobileye)](https://www.mobileye.com/blog/eyeq6-system-on-chip/)
[8] [Mobileye EyeQ6 Lite Launches (Mobileye IR)](https://ir.mobileye.com/news-releases/news-release-details/mobileye-eyeq6-lite-launches-speed-adas-upgrades-worldwide)
[9] [An Exposure-Time-Aligned Primary-Path Architecture for Autonomous-Driving ECUs (arXiv 2606.10856)](https://arxiv.org/abs/2606.10856)
[10] [Measuring What Actually Matters in Real-Time ADAS Perception (Fixstars)](https://blog.us.fixstars.com/measuring-what-actually-matters-in-real-time-adas-perception/)

**추가 참고**:
- [Tesla FSD Processes 1 Million Pixels Per Millisecond (Basenor)](https://www.basenor.com/blogs/news/tesla-fsd-processes-1-million-pixels-per-millisecond-how)
- [NVIDIA Orin SoC Analysis for ADAS (Aivon)](https://www.aivon.com/blog/automotive-electronics/nvidia-orin-soc-analysis-for-adas/)
- [The fast lane to higher levels of autonomy with EyeQ6 (Mobileye)](https://www.mobileye.com/blog/the-fast-lane-to-higher-levels-of-autonomy-with-the-eyeq6-soc/)

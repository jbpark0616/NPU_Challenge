# Fusion 관점의 SDC Detection 고민 (팀 공유용)

작성일: 2026-07-24

> Fusion 이 만드는 문제를 4 개 FT 코어에 얹은 ABFT 로직으로 처리할 수 있을지, 그리고 그게 DMR 보다 나은 커버리지를 낼 수 있을지에 대한 고민.

---

## 1. 하드웨어 관찰

Tensix compute pipeline:
```
UNPACK → FPU (matmul, DEST 누산) → SFPU (element-wise, DEST in-place) → PACK
                  ↑                        ↑
                  └── DEST register 를 공유
```

- **FPU** = matmul/conv 담당, 결과가 DEST 에 누산
- **SFPU** = ReLU/pool 등 담당, DEST 를 in-place 로 변환
- Fused conv+ReLU = 하나의 kernel 안에서 두 유닛이 DEST 공유하며 파이프라이닝

즉 fusion 은 소프트웨어 편의가 아니라 **하드웨어 구조가 만든 것**.

---

## 2. BOS ResNet50 fusion 패턴 (실측 확인)

`bos_model_copy/resnet50/tt/` 확인 결과:

### Compile-time
- **BN 은 conv weight/bias 로 folding** (`fold_batch_norm2d_into_conv2d`)
- Runtime 에 BN 연산 자체가 없음

### Runtime fusion (`ttnn.Conv2dConfig(activation=...)`)
- **conv1, conv2**: `activation=RELU` → **fused**
- **conv3**: activation 없음 → **unfused** (거의 유일한 예외)
- **Residual add**: `add_(activations=[RELU])` → **fused**

### Op 분포 (대략)
- 약 95% 가 fused 형태 (conv+ReLU 또는 Add+ReLU)
- 순수 선형 op 은 conv3 와 최종 FC 정도

---

## 3. Fusion 이 만드는 문제

Classical ABFT row/col checksum invariant:
```
sum(row i of C) = A[i,:] · sum(rows of B)     (C = A × B 일 때)
```

ReLU 통과하면:
```
C_relu[i,j] = max(0, C[i,j])
sum(row i of C_relu) ≠ sum(row i of C)   (음수 clip 됨)
```

→ **선형 invariant 가 fused output 에 성립하지 않음.**

BOS 는 대부분 op 이 fused 라서 classical ABFT 를 그대로 적용할 수 없음.

---

## 4. 고민 중인 방향

**4 코어 FT 를 fused op 에 맞춘 ABFT 로직으로 채우고, DMR 대비 나은 커버리지를 낼 수 있을지 검토 중.**

### DMR 기준선
- 4 코어로 16 코어 duplicate → 25% instantaneous coverage
- Rotation 하면 4-frame 뒤에 100% 이지만 detection latency 큼
- Common-mode (같은 데이터 경로) 오류 못 잡음

### Fusion-aware ABFT 아이디어
- Fused output 에 성립하는 invariant 를 새로 유도
  - 예: positive-only checksum (ReLU 이후 살아남는 mass 로 만든 sum)
  - 예: magnitude bound (`|out| ≤ ...`)
- 4 코어에 이 로직을 얹으면 op 당 O(N) 부담 → 리소스 여유 있음
- Instantaneous 로 layer 별 검증 가능
- DMR 보다 리소스 적게 쓰면서 커버리지 넓을 것으로 기대

### 왜 유의미할 가능성이 있나
- 기존 ABFT 문헌은 순수 선형 op 가정 → fused op 은 gap
- Modern spatial NPU 는 fusion 이 표준
- BOS 실측 상 95% 가 fused → gap 이 실제 자동차 워크로드에서 크다

---

## 5. 남은 고민

- Fused op 을 위한 새 invariant 를 실제로 유도할 수 있나 (positive-only 는 bound 만, 정확한 등식 미해결)
- 유도된 invariant 의 실측 커버리지 (fault injection 결과) 가 DMR 을 정말 이기는가
- Fusion 을 그대로 두고 검증하는 게 정답인지, 아니면 detector 도 fused op 을 재실행해야 하는지
- 4:16 리소스 비율에서 커버리지 상한이 어디까지인지

---

## 6. 참고 파일

- `bos_model_copy/resnet50/tt/ttnn_functional_resnet50.py`: fusion 실제 사용 (`activation=RELU`)
- `bos_model_copy/resnet50/tt/custom_preprocessing.py`: BN folding

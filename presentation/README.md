# Presentation Materials

최종 발표 보고서와 README에 사용한 시각 자료를 보관한다.

| 위치 | 내용 |
|---|---|
| [`NPU_challenge_Final_Report.pdf`](NPU_challenge_Final_Report.pdf) | 2026-08-21 최종 발표 보고서 |
| [`figures/`](figures/) | architecture, Conv data movement, core remapping, planner, MILP 결과 SVG와 보조 이미지 |

대표 figure:

- `A0_NPU_arch.svg`: Eagle-N A0 architecture와 worker grid
- `topleft_hardcoding.svg`: 고정된 top-left placement 문제
- `height_conv.svg`, `1x1_or_block_conv.svg`: Conv sharding과 data movement
- `core_remapping_overview.svg`: placement remapping 개요
- `core_remapping_mcast.svg`: segmented rectangular multicast
- `remapping_planner.svg`, `remappin_panner_timeline.svg`: logical-to-physical mapping과 placement lifetime
- `segment_scope_ablation.svg`: MILP segment-scope ablation

최종 수치의 source data와 생성 script는 해당 실험 폴더에 둔다.

- [`../data_movement/conv_core_replacement/policy/figures/`](../data_movement/conv_core_replacement/policy/figures/): MILP 및 validated latency figure
- [`../study_note/aging_simulation/plots/`](../study_note/aging_simulation/plots/): reliability simulation plot

프로젝트 전체 요약과 각 결과의 주장 범위는 [최상위 README](../README.md)를 참고한다.

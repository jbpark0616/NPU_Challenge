이 NPU에 aging-aware / wear-leveling 코어 매핑을 구현할 수 있는지 평가 중이야. 구체적으로 각 op이 쓰는 물리 코어를 추론 실행마다 rotation 해서 트랜지스터 stress를 코어 전체에 고르게 분산시키는 게 목표야. 본격 착수 전에 현재 tt-metal/ttnn 스택에서 코어 배치가 어떻게 결정되는지 정확히 알아야 해.

코드베이스를 조사해서, 파일 경로·함수명·코드 스니펫을 증거로 아래에 답해줘:

compute op(ttnn.conv2d, matmul 등)의 코어 할당/sharding이 어디서 결정돼? 몇 개 코어를, 그리고 어느 코어를 쓸지 정하는 파일/함수는?
이 배치가 결정론적(deterministic)이야? 같은 op·같은 입력 shape이면 매 실행마다 항상 같은 물리 코어에 매핑돼? 아니면 randomization/변동이 있어?
배치를 override할 수 있어? Conv2dConfig, core grid, shard spec, CoreRangeSet 등으로 특정 op을 특정 코어 집합에 강제할 수 있나? 그 API/config와 어디서 소비되는지 보여줘.
다음 개념이 코드에 존재해? — per-core 활용률 추적, wear-leveling, lifetime, aging, 또는 시간에 걸쳐 코어 간 일을 rotation하는 것. (없을 것 같지만 확인.)
"op이 쓰는 코어 집합을 실행마다 rotation"(temporal wear-leveling)을 구현하려면 자연스러운 hook 지점이 어디야? 뭘 수정해야 하고, 그게 성능 autotuner / sharding 로직과 충돌하나?

딜리버러블: 1~5에 대한 짧은 요약 + 핵심 파일 경로/함수 목록 + 5번(temporal rotation)이 현재 아키텍처에서 얼마나 어려운지 솔직한 평가.
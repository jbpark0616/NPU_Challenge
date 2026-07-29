"""Stage-split boundary artifact op filter.

각 stage 진입/종료 시 발생하는 layout/sharding 전환 op 은 real ResNet50 forward pass 에는
없는 methodology artifact. Aggregation 에서 real op 통계와 분리하기 위해 이름으로 라벨링.

참고: [[project-stage-split-boundary-artifact]]
"""
from __future__ import annotations

BOUNDARY_OP_NAMES: frozenset[str] = frozenset({
    "InterleavedToShardedDeviceOperation",
    "ShardedToInterleavedDeviceOperation",
    "TilizeWithValPadding",
    "Tilize",
    "UntilizeWithUnpadding",
    "Untilize",
    "ReshardDeviceOperation",
    "MoveDeviceOperation",
})


def is_boundary(op_name: str) -> bool:
    return op_name in BOUNDARY_OP_NAMES

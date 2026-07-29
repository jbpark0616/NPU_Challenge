"""A0 (Eagle-N translated) 6×5 격자 위 X-Y dimension-ordered NoC routing.

Blackhole `unicastRoute` (tt-npe/tt_npe/cpp/include/device_models/blackhole.hpp:331)
로직을 그대로 이식. 격자 크기만 6×5. Torus wrap 존재.

NOC 0: col 먼저 (+1 EAST) → row (+1 SOUTH). 둘 다 wrap.
NOC 1: row 먼저 (-1 NORTH) → col (-1 WEST). 둘 다 wrap.

Link 식별: (row, col, direction). 즉 "이 셀에서 이 방향으로 나가는 링크".
- NOC0_EAST 는 (r, c) → (r, (c+1) % COLS)
- NOC0_SOUTH 는 (r, c) → ((r+1) % ROWS, c)
- NOC1_NORTH 는 (r, c) → ((r-1) % ROWS, c)
- NOC1_WEST 는 (r, c) → (r, (c-1) % COLS)

우리 raw noc_trace 는 sx = col, sy = row (blackhole.hpp {device, row, col} 와 일치).
"""
from __future__ import annotations

from typing import Iterator

# A0 translated grid (관측 확정)
GRID_ROWS = 5  # y=0..4
GRID_COLS = 6  # x=0..5


# Link direction constants
NOC0_EAST = "NOC0_EAST"
NOC0_SOUTH = "NOC0_SOUTH"
NOC1_NORTH = "NOC1_NORTH"
NOC1_WEST = "NOC1_WEST"

ALL_LINK_TYPES = (NOC0_EAST, NOC0_SOUTH, NOC1_NORTH, NOC1_WEST)


LinkID = tuple[int, int, str]  # (row, col, dir)


def _wrap(v: int, n: int) -> int:
    return v % n


def route_noc0(sr: int, sc: int, dr: int, dc: int) -> list[LinkID]:
    """NOC 0: col 먼저 (+1 EAST wrap), 그 다음 row (+1 SOUTH wrap)."""
    route: list[LinkID] = []
    r, c = sr, sc
    guard = 0
    while (r, c) != (dr, dc):
        if c != dc:
            route.append((r, c, NOC0_EAST))
            c = _wrap(c + 1, GRID_COLS)
        elif r != dr:
            route.append((r, c, NOC0_SOUTH))
            r = _wrap(r + 1, GRID_ROWS)
        guard += 1
        if guard > GRID_ROWS * GRID_COLS * 2:
            raise RuntimeError(f"route_noc0 stuck: ({sr},{sc}) → ({dr},{dc})")
    return route


def route_noc1(sr: int, sc: int, dr: int, dc: int) -> list[LinkID]:
    """NOC 1: row 먼저 (-1 NORTH wrap), 그 다음 col (-1 WEST wrap)."""
    route: list[LinkID] = []
    r, c = sr, sc
    guard = 0
    while (r, c) != (dr, dc):
        if r != dr:
            route.append((r, c, NOC1_NORTH))
            r = _wrap(r - 1, GRID_ROWS)
        elif c != dc:
            route.append((r, c, NOC1_WEST))
            c = _wrap(c - 1, GRID_COLS)
        guard += 1
        if guard > GRID_ROWS * GRID_COLS * 2:
            raise RuntimeError(f"route_noc1 stuck: ({sr},{sc}) → ({dr},{dc})")
    return route


def route(sr: int, sc: int, dr: int, dc: int, noc: str) -> list[LinkID]:
    """noc = 'NOC_0' | 'NOC_1' (raw JSON 필드 값)."""
    if noc == "NOC_0":
        return route_noc0(sr, sc, dr, dc)
    if noc == "NOC_1":
        return route_noc1(sr, sc, dr, dc)
    # 알 수 없는 noc — 빈 경로 (endpoint-only 로 처리하고 log)
    return []


def link_endpoints(link: LinkID) -> tuple[tuple[int, int], tuple[int, int]]:
    """링크의 (from_cell, to_cell) 반환. 시각화 arrow 그릴 때 사용."""
    r, c, d = link
    if d == NOC0_EAST:
        return (r, c), (r, _wrap(c + 1, GRID_COLS))
    if d == NOC0_SOUTH:
        return (r, c), (_wrap(r + 1, GRID_ROWS), c)
    if d == NOC1_NORTH:
        return (r, c), (_wrap(r - 1, GRID_ROWS), c)
    if d == NOC1_WEST:
        return (r, c), (r, _wrap(c - 1, GRID_COLS))
    raise ValueError(f"unknown link direction: {d}")


def iter_all_links() -> Iterator[LinkID]:
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            for d in ALL_LINK_TYPES:
                yield (r, c, d)


# ============================================================
# Self-test / smoke test
# ============================================================
def _test():
    print(f"grid: {GRID_ROWS} rows × {GRID_COLS} cols = {GRID_ROWS * GRID_COLS} cells")
    print(f"total links: {sum(1 for _ in iter_all_links())}")
    print()

    # 예 1: (row=2, col=1) → (row=2, col=4) via NOC 1 (screenshot 예시)
    r = route_noc1(2, 1, 2, 4)
    print(f"NOC1  (r=2,c=1) → (r=2,c=4):  {len(r)} hops")
    for h in r:
        print(f"   {h}")
    print()

    # 예 2: 같은 transfer via NOC 0
    r = route_noc0(2, 1, 2, 4)
    print(f"NOC0  (r=2,c=1) → (r=2,c=4):  {len(r)} hops")
    for h in r:
        print(f"   {h}")
    print()

    # 예 3: op235 첫 transfer (0,0) → (3,1) via NOC 1
    r = route_noc1(0, 0, 3, 1)
    print(f"NOC1  (r=0,c=0) → (r=3,c=1):  {len(r)} hops")
    for h in r:
        print(f"   {h}")
    print()

    # 예 4: 대각선 endpoint 확인
    r = route_noc0(0, 0, 4, 5)
    print(f"NOC0  (r=0,c=0) → (r=4,c=5):  {len(r)} hops")


if __name__ == "__main__":
    _test()

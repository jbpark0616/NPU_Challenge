"""Rotation simulation Tier 1: anchor shift + round-robin policy.

Assumptions:
  - Translation symmetry: shifting an op's anchor translates its activity
    pattern rigidly. Internal mcast/compute structure preserved.
  - Strict-fit: op shape must entirely fit within compute grid.
  - Wrap: NOT supported (validated in source verified in BASELINE_NOTE §6).

Input:  matrix.csv (per-op × per-core injected/consumed bytes)
Output: rotated per-core activity CSV, baseline-vs-rotation comparison.

Grid convention:
  - Compute cores: y in {0,1,2,4}, x in {0..4} = 20 cores (§2.2 of note)
  - Logical y_map: physical y=4 <-> logical y=3 (DRAM row y=3 skipped)
  - Anchor shift works in LOGICAL space (0..3, 0..3) to avoid DRAM row.
"""
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

REPO = Path('C:/NPU_Challenge')
MATRIX_CSV = REPO / 'study_note/aging_simulation/NoC_analysis/outputs/matrix.csv'
OUT_DIR    = REPO / 'study_note/aging_simulation/plots'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Physical -> logical y mapping (skip DRAM row y=3)
PHYS_TO_LOG_Y = {0: 0, 1: 1, 2: 2, 4: 3}
LOG_TO_PHYS_Y = {v: k for k, v in PHYS_TO_LOG_Y.items()}
COMPUTE_PHYS_Y = set(PHYS_TO_LOG_Y.keys())
COMPUTE_X = set(range(5))
LOGICAL_GRID_W = 5   # x: 0..4
LOGICAL_GRID_H = 4   # y: 0..3 (logical)


def load_per_op_activity():
    """Return {op_id: {(logical_x, logical_y): (inj, con)}} restricted to compute cores."""
    df = pd.read_csv(MATRIX_CSV)
    df = df[df['is_boundary'] == 0]
    df = df[df['sx'].isin(COMPUTE_X) & df['sy'].isin(COMPUTE_PHYS_Y)]
    df['ly'] = df['sy'].map(PHYS_TO_LOG_Y)

    ops = {}
    for (stage, rhid, op_name), grp in df.groupby(['stage', 'run_host_id', 'op_name']):
        op_id = (int(stage), int(rhid), op_name)
        activity = {}
        for _, r in grp.iterrows():
            lx, ly = int(r['sx']), int(r['ly'])
            inj = int(r['injected_bytes'])
            con = int(r['consumed_bytes'])
            if inj > 0 or con > 0:
                activity[(lx, ly)] = (inj, con)
        if activity:
            ops[op_id] = activity
    return ops


def compute_bbox(activity):
    """Return (min_x, min_y, max_x, max_y) in logical coords."""
    xs = [x for (x, _) in activity]
    ys = [y for (_, y) in activity]
    return min(xs), min(ys), max(xs), max(ys)


def valid_anchors(shape_w, shape_h, grid_w=LOGICAL_GRID_W, grid_h=LOGICAL_GRID_H):
    """Strict-fit anchor positions for given shape on grid."""
    return [(x, y)
            for y in range(grid_h - shape_h + 1)
            for x in range(grid_w - shape_w + 1)]


def rotate_activity(activity, origin_x, origin_y, target_x, target_y):
    """Translation-symmetric shift: move each core by (dx, dy)."""
    dx = target_x - origin_x
    dy = target_y - origin_y
    return {(cx + dx, cy + dy): v for (cx, cy), v in activity.items()}


def rotation_schedule(ops, policy='round_robin'):
    """Return {op_id: chosen_anchor}."""
    counter = defaultdict(int)   # shape -> cycle index
    schedule = {}
    for op_id, activity in ops.items():
        min_x, min_y, max_x, max_y = compute_bbox(activity)
        w = max_x - min_x + 1
        h = max_y - min_y + 1
        anchors = valid_anchors(w, h)
        if not anchors:
            # Op larger than grid — shouldn't happen with valid data
            chosen = (min_x, min_y)
        elif policy == 'round_robin':
            shape_key = (w, h)
            idx = counter[shape_key] % len(anchors)
            counter[shape_key] += 1
            chosen = anchors[idx]
        else:
            raise ValueError(f'Unknown policy: {policy}')
        schedule[op_id] = {
            'shape': (w, h),
            'origin': (min_x, min_y),
            'chosen': chosen,
            'valid_count': len(anchors),
        }
    return schedule


def aggregate_activity(ops, schedule):
    """Return per-core (injected, consumed) aggregate under given schedule."""
    per_core = defaultdict(lambda: [0, 0])
    for op_id, activity in ops.items():
        info = schedule[op_id]
        rotated = rotate_activity(activity,
                                  info['origin'][0], info['origin'][1],
                                  info['chosen'][0], info['chosen'][1])
        for (lx, ly), (inj, con) in rotated.items():
            per_core[(lx, ly)][0] += inj
            per_core[(lx, ly)][1] += con
    return {k: tuple(v) for k, v in per_core.items()}


def to_dataframe(per_core, label):
    rows = []
    for (lx, ly), (inj, con) in per_core.items():
        rows.append({
            'lx': lx, 'ly': ly,
            'phys_y': LOG_TO_PHYS_Y.get(ly, -1),
            'injected_bytes': inj, 'consumed_bytes': con,
            'total_noc': inj + con,
            'source': label,
        })
    return pd.DataFrame(rows).sort_values(['ly', 'lx']).reset_index(drop=True)


def compute_alpha(per_core, axis):
    """Return normalized alpha (worst=1)."""
    if axis == 'niu':
        vals = np.array([inj + con for (inj, con) in per_core.values()], dtype=float)
    elif axis == 'sram':
        vals = np.array([con for (inj, con) in per_core.values()], dtype=float)
    elif axis == 'injected':
        vals = np.array([inj for (inj, con) in per_core.values()], dtype=float)
    else:
        raise ValueError(axis)
    m = vals.max()
    if m == 0:
        return vals
    return (vals / m).clip(min=1e-6)


def spread(per_core, axis):
    a = compute_alpha(per_core, axis)
    return a.max() / a.min()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print('Loading per-op activity from matrix.csv ...')
    ops = load_per_op_activity()
    print(f'  {len(ops)} ops with compute-core activity')

    # Baseline: original anchors (no rotation)
    baseline_schedule = {
        op_id: {'shape': None, 'origin': compute_bbox(act)[:2],
                'chosen': compute_bbox(act)[:2], 'valid_count': 1}
        for op_id, act in ops.items()
    }
    baseline_per_core = aggregate_activity(ops, baseline_schedule)
    print(f'\nBaseline per-core activity:')
    print(f'  NIU spread : {spread(baseline_per_core, "niu"):.2f}x')
    print(f'  SRAM spread: {spread(baseline_per_core, "sram"):.2f}x')

    # Rotation: round-robin anchor
    schedule = rotation_schedule(ops, policy='round_robin')
    rotated_per_core = aggregate_activity(ops, schedule)
    print(f'\nRotation (round-robin) per-core activity:')
    print(f'  NIU spread : {spread(rotated_per_core, "niu"):.2f}x')
    print(f'  SRAM spread: {spread(rotated_per_core, "sram"):.2f}x')

    # Distribution of rotation freedom
    shape_freedom = defaultdict(list)
    for op_id, info in schedule.items():
        if info['shape']:
            shape_freedom[info['shape']].append(info['valid_count'])
    print(f'\nOp shape → valid anchor count:')
    for shape, counts in sorted(shape_freedom.items(), key=lambda x: -x[0][0] * x[0][1]):
        w, h = shape
        n_ops = len(counts)
        anchors = counts[0]
        print(f'  {w}x{h}: {n_ops:3d} ops × {anchors} anchors')

    # Write CSVs
    baseline_df = to_dataframe(baseline_per_core, 'baseline')
    rotated_df  = to_dataframe(rotated_per_core, 'rotation_rr')
    combined = pd.concat([baseline_df, rotated_df], ignore_index=True)
    out_csv = OUT_DIR / 'rotation_tier1_per_core.csv'
    combined.to_csv(out_csv, index=False)
    print(f'\nWrote {out_csv}')

    # Comparison summary
    print(f'\n=== Baseline vs Rotation-RR (Tier 1) ===')
    for axis in ('niu', 'sram'):
        b = spread(baseline_per_core, axis)
        r = spread(rotated_per_core, axis)
        print(f'  {axis:5s} spread : {b:6.2f}x → {r:6.2f}x   (improvement {b/r:.2f}x)')


if __name__ == '__main__':
    main()

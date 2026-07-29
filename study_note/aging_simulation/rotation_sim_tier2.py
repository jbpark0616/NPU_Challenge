"""Rotation simulation Tier 2: anchor shift + reflection + round-robin policy.

Extends Tier 1 by adding reflection for block-sharded 4x4 ops:
  - TL (default): weight sender = top row, activation sender = left col
  - TR: activation sender = right col (weight sender still top)
  - BL: weight sender = bottom row (activation sender still left)
  - BR: both sender edges flipped

For a 4x4 shape on 5x4 grid:
  - Anchor positions: 2 (x=0 or x=1)
  - Reflection modes: 4 (TL / TR / BL / BR)
  - Total configs: 2 x 4 = 8

Reflection is applied ONLY to block-sharded ops (identified by shape 4x4
which corresponds to CORE_SET_16 used in Stage 3-4). Height-sharded ops
have a single sender core so reflection is not meaningful.
"""
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

REPO = Path('C:/NPU_Challenge')
MATRIX_CSV = REPO / 'study_note/aging_simulation/NoC_analysis/outputs/matrix.csv'
OUT_DIR    = REPO / 'study_note/aging_simulation/plots'
OUT_DIR.mkdir(parents=True, exist_ok=True)

PHYS_TO_LOG_Y = {0: 0, 1: 1, 2: 2, 4: 3}
LOG_TO_PHYS_Y = {v: k for k, v in PHYS_TO_LOG_Y.items()}
COMPUTE_PHYS_Y = set(PHYS_TO_LOG_Y.keys())
COMPUTE_X = set(range(5))
LOGICAL_GRID_W = 5
LOGICAL_GRID_H = 4

# Shapes considered block-sharded (candidates for reflection).
# 4x4 = CORE_SET_16 (Layer 3-4 block-sharded confirmed via tracy grep)
BLOCK_SHAPES = {(4, 4)}


def load_per_op_activity():
    df = pd.read_csv(MATRIX_CSV)
    df = df[df['is_boundary'] == 0]
    df = df[df['sx'].isin(COMPUTE_X) & df['sy'].isin(COMPUTE_PHYS_Y)]
    df['ly'] = df['sy'].map(PHYS_TO_LOG_Y)
    ops = {}
    for (stage, rhid, op_name), grp in df.groupby(['stage', 'run_host_id', 'op_name']):
        op_id = (int(stage), int(rhid), op_name)
        act = {}
        for _, r in grp.iterrows():
            lx, ly = int(r['sx']), int(r['ly'])
            inj, con = int(r['injected_bytes']), int(r['consumed_bytes'])
            if inj > 0 or con > 0:
                act[(lx, ly)] = (inj, con)
        if act:
            ops[op_id] = act
    return ops


def compute_bbox(activity):
    xs = [x for (x, _) in activity]
    ys = [y for (_, y) in activity]
    return min(xs), min(ys), max(xs), max(ys)


def valid_anchors(w, h, grid_w=LOGICAL_GRID_W, grid_h=LOGICAL_GRID_H):
    return [(x, y)
            for y in range(grid_h - h + 1)
            for x in range(grid_w - w + 1)]


def apply_reflection(local_activity, mode, w, h):
    """Reflect activity within local shape (0..w-1, 0..h-1).
      TL: no flip
      TR: flip x
      BL: flip y
      BR: flip both
    """
    if mode == 'TL':
        return dict(local_activity)
    result = {}
    for (x, y), v in local_activity.items():
        nx = (w - 1 - x) if mode in ('TR', 'BR') else x
        ny = (h - 1 - y) if mode in ('BL', 'BR') else y
        result[(nx, ny)] = v
    return result


def rotation_schedule_tier2(ops, policy='round_robin'):
    """Return {op_id: {shape, origin, chosen_anchor, reflection, valid_count}}."""
    counter = defaultdict(int)   # (shape, is_block) -> cycle idx
    schedule = {}
    for op_id, activity in ops.items():
        min_x, min_y, max_x, max_y = compute_bbox(activity)
        w = max_x - min_x + 1
        h = max_y - min_y + 1
        anchors = valid_anchors(w, h)
        is_block = (w, h) in BLOCK_SHAPES
        modes = ['TL', 'TR', 'BL', 'BR'] if is_block else ['TL']
        # Combined config space
        configs = [(a, m) for a in anchors for m in modes]

        if not configs:
            chosen_anchor, chosen_mode = (min_x, min_y), 'TL'
        else:
            key = ((w, h), is_block)
            idx = counter[key] % len(configs)
            counter[key] += 1
            chosen_anchor, chosen_mode = configs[idx]

        schedule[op_id] = {
            'shape': (w, h),
            'origin': (min_x, min_y),
            'chosen_anchor': chosen_anchor,
            'reflection': chosen_mode,
            'is_block': is_block,
            'valid_config_count': len(configs),
        }
    return schedule


def rotate_reflect_translate(activity, origin, target_anchor, reflection, w, h):
    """Full transform: local origin -> reflect -> translate to anchor."""
    ox, oy = origin
    ax, ay = target_anchor
    # Step 1: to local
    local = {(x - ox, y - oy): v for (x, y), v in activity.items()}
    # Step 2: reflect within local
    reflected = apply_reflection(local, reflection, w, h)
    # Step 3: translate
    return {(x + ax, y + ay): v for (x, y), v in reflected.items()}


def aggregate(ops, schedule):
    per_core = defaultdict(lambda: [0, 0])
    for op_id, activity in ops.items():
        info = schedule[op_id]
        w, h = info['shape']
        moved = rotate_reflect_translate(
            activity, info['origin'], info['chosen_anchor'],
            info['reflection'], w, h)
        for (lx, ly), (inj, con) in moved.items():
            per_core[(lx, ly)][0] += inj
            per_core[(lx, ly)][1] += con
    return {k: tuple(v) for k, v in per_core.items()}


def alpha(per_core, axis):
    if axis == 'niu':
        vals = np.array([i + c for (i, c) in per_core.values()], dtype=float)
    elif axis == 'sram':
        vals = np.array([c for (i, c) in per_core.values()], dtype=float)
    else:
        raise ValueError(axis)
    m = vals.max()
    if m == 0:
        return vals
    return (vals / m).clip(min=1e-6)


def spread(per_core, axis):
    a = alpha(per_core, axis)
    return a.max() / a.min()


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


def main():
    ops = load_per_op_activity()
    print(f'Loaded {len(ops)} ops')

    # Baseline: chosen_anchor = origin (no shift, no reflection)
    baseline_sched = {}
    for op_id, act in ops.items():
        mnx, mny, mxx, mxy = compute_bbox(act)
        baseline_sched[op_id] = {
            'shape': (mxx - mnx + 1, mxy - mny + 1),
            'origin': (mnx, mny),
            'chosen_anchor': (mnx, mny),
            'reflection': 'TL',
            'is_block': False,
            'valid_config_count': 1,
        }
    base_pc = aggregate(ops, baseline_sched)

    # Tier 2
    sched = rotation_schedule_tier2(ops)
    rot_pc = aggregate(ops, sched)

    # Config count summary
    print('\n=== Rotation config space per shape ===')
    shape_configs = defaultdict(list)
    for op_id, info in sched.items():
        shape_configs[(info['shape'], info['is_block'])].append(info['valid_config_count'])
    for (shape, is_blk), configs in sorted(shape_configs.items(), key=lambda x: -x[0][0][0] * x[0][0][1]):
        w, h = shape
        n_ops = len(configs)
        n_cfg = configs[0]
        label = 'block' if is_blk else 'height/other'
        print(f'  {w}x{h} [{label}]: {n_ops:3d} ops × {n_cfg:2d} configs')

    print('\n=== Baseline vs Tier 1 (shift only) vs Tier 2 (shift + reflect) ===')
    # Reuse Tier 1 schedule
    from rotation_sim_tier1 import rotation_schedule as t1_schedule
    t1_sched = t1_schedule(ops)
    # Add reflection field to make compatible
    for k in t1_sched:
        t1_sched[k]['reflection'] = 'TL'
        t1_sched[k]['chosen_anchor'] = t1_sched[k]['chosen']
    t1_pc = aggregate(ops, t1_sched)

    for axis in ('niu', 'sram'):
        b  = spread(base_pc, axis)
        t1 = spread(t1_pc,  axis)
        t2 = spread(rot_pc, axis)
        print(f'  {axis:5s}:  base={b:6.2f}x  |  Tier1={t1:6.2f}x  ({b/t1:.2f}×↓)  |  Tier2={t2:6.2f}x  ({b/t2:.2f}×↓)')

    # Write CSV
    df = pd.concat([
        to_dataframe(base_pc, 'baseline'),
        to_dataframe(t1_pc, 'tier1'),
        to_dataframe(rot_pc, 'tier2'),
    ], ignore_index=True)
    out = OUT_DIR / 'rotation_tier2_per_core.csv'
    df.to_csv(out, index=False)
    print(f'\nWrote {out}')


if __name__ == '__main__':
    main()

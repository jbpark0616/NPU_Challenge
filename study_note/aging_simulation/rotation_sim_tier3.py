"""Rotation simulation Tier 3: aging-aware greedy policy.

Extends Tier 2 by replacing round-robin with a state-aware policy:
per-core aging state is tracked, and for each op we pick the config
that minimizes the resulting worst-core stress.

Policy: argmin_{config} max_{core} (aging_state[core] + activity_after_config[core])

State variable: total_noc = injected + consumed bytes (proxy for combined stress)

Config space per op (same as Tier 2):
  - Height/other shapes: anchors × 1 reflection (TL only)
  - Block 4x4 shapes:    anchors × 4 reflections (TL/TR/BL/BR)
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
GRID_W = 5
GRID_H = 4

BLOCK_SHAPES = {(4, 4)}   # 4x4 = CORE_SET_16 (block-sharded, has 2 mcast axes)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_per_op_activity():
    df = pd.read_csv(MATRIX_CSV)
    df = df[df['is_boundary'] == 0]
    df = df[df['sx'].isin(COMPUTE_X) & df['sy'].isin(COMPUTE_PHYS_Y)]
    df['ly'] = df['sy'].map(PHYS_TO_LOG_Y)
    ops = []
    for (stage, rhid, op_name), grp in df.groupby(['stage', 'run_host_id', 'op_name']):
        activity = {}
        for _, r in grp.iterrows():
            lx, ly = int(r['sx']), int(r['ly'])
            inj, con = int(r['injected_bytes']), int(r['consumed_bytes'])
            if inj > 0 or con > 0:
                activity[(lx, ly)] = (inj, con)
        if activity:
            ops.append({
                'id': (int(stage), int(rhid), op_name),
                'stage': int(stage),
                'rhid': int(rhid),
                'op_name': op_name,
                'activity': activity,
            })
    ops.sort(key=lambda o: (o['stage'], o['rhid']))   # deterministic order
    return ops


def compute_bbox(activity):
    xs = [x for (x, _) in activity]
    ys = [y for (_, y) in activity]
    return min(xs), min(ys), max(xs), max(ys)


# ---------------------------------------------------------------------------
# Config space
# ---------------------------------------------------------------------------
def valid_anchors(w, h):
    return [(x, y)
            for y in range(GRID_H - h + 1)
            for x in range(GRID_W - w + 1)]


def valid_configs(w, h, is_block):
    anchors = valid_anchors(w, h)
    modes = ['TL', 'TR', 'BL', 'BR'] if is_block else ['TL']
    return [(a, m) for a in anchors for m in modes]


def apply_reflection_local(local_activity, mode, w, h):
    if mode == 'TL':
        return dict(local_activity)
    result = {}
    for (x, y), v in local_activity.items():
        nx = (w - 1 - x) if mode in ('TR', 'BR') else x
        ny = (h - 1 - y) if mode in ('BL', 'BR') else y
        result[(nx, ny)] = v
    return result


def transform_activity(activity, origin, target_anchor, reflection, w, h):
    ox, oy = origin
    ax, ay = target_anchor
    local = {(x - ox, y - oy): v for (x, y), v in activity.items()}
    reflected = apply_reflection_local(local, reflection, w, h)
    return {(x + ax, y + ay): v for (x, y), v in reflected.items()}


# ---------------------------------------------------------------------------
# Aging-aware scheduler
# ---------------------------------------------------------------------------
def score_worst_core(state, moved_activity):
    """After placing this op with given transform, what's the worst-core NoC total?"""
    # Union of state + this op's contribution
    worst = 0
    # Consider all cores in state OR touched by op
    touched = set(state.keys()) | set(moved_activity.keys())
    for core in touched:
        base = state.get(core, 0)
        add = 0
        if core in moved_activity:
            inj, con = moved_activity[core]
            add = inj + con
        worst = max(worst, base + add)
    return worst


def schedule_aging_aware(ops):
    """Greedy: pick config minimizing worst-core after each op."""
    state = defaultdict(int)   # per-core cumulative (inj + con)
    schedule = {}
    for op in ops:
        activity = op['activity']
        mnx, mny, mxx, mxy = compute_bbox(activity)
        w = mxx - mnx + 1
        h = mxy - mny + 1
        is_block = (w, h) in BLOCK_SHAPES
        configs = valid_configs(w, h, is_block)

        if not configs:
            configs = [((mnx, mny), 'TL')]

        best_config = None
        best_worst = None
        for (anchor, mode) in configs:
            moved = transform_activity(activity, (mnx, mny), anchor, mode, w, h)
            worst = score_worst_core(state, moved)
            if best_worst is None or worst < best_worst:
                best_worst = worst
                best_config = (anchor, mode)

        # Commit
        anchor, mode = best_config
        moved = transform_activity(activity, (mnx, mny), anchor, mode, w, h)
        for core, (inj, con) in moved.items():
            state[core] += inj + con

        schedule[op['id']] = {
            'shape': (w, h),
            'origin': (mnx, mny),
            'chosen_anchor': anchor,
            'reflection': mode,
            'is_block': is_block,
            'valid_config_count': len(configs),
        }
    return schedule


def aggregate(ops, schedule):
    per_core = defaultdict(lambda: [0, 0])
    for op in ops:
        info = schedule[op['id']]
        w, h = info['shape']
        moved = transform_activity(op['activity'], info['origin'],
                                   info['chosen_anchor'], info['reflection'],
                                   w, h)
        for (lx, ly), (inj, con) in moved.items():
            per_core[(lx, ly)][0] += inj
            per_core[(lx, ly)][1] += con
    return {k: tuple(v) for k, v in per_core.items()}


def spread(per_core, axis):
    if axis == 'niu':
        vals = np.array([i + c for (i, c) in per_core.values()], dtype=float)
    elif axis == 'sram':
        vals = np.array([c for (i, c) in per_core.values()], dtype=float)
    else:
        raise ValueError(axis)
    m_max = vals.max()
    m_min = vals[vals > 0].min() if any(vals > 0) else 1
    return m_max / m_min


def worst_core(per_core, axis):
    if axis == 'niu':
        vals = np.array([i + c for (i, c) in per_core.values()], dtype=float)
    elif axis == 'sram':
        vals = np.array([c for (i, c) in per_core.values()], dtype=float)
    else:
        raise ValueError(axis)
    return vals.max()


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


# ---------------------------------------------------------------------------
# Compare all tiers
# ---------------------------------------------------------------------------
def main():
    ops = load_per_op_activity()
    print(f'Loaded {len(ops)} ops (sorted by execution order)')

    # Baseline (no rotation)
    baseline_sched = {}
    for op in ops:
        mnx, mny, mxx, mxy = compute_bbox(op['activity'])
        baseline_sched[op['id']] = {
            'shape': (mxx - mnx + 1, mxy - mny + 1),
            'origin': (mnx, mny),
            'chosen_anchor': (mnx, mny),
            'reflection': 'TL',
            'is_block': False,
            'valid_config_count': 1,
        }
    base_pc = aggregate(ops, baseline_sched)

    # Tier 1: shift + round-robin (no reflection)
    from rotation_sim_tier1 import rotation_schedule as t1_rr
    t1_sched = t1_rr({op['id']: op['activity'] for op in ops})
    for k in t1_sched:
        t1_sched[k]['reflection'] = 'TL'
        t1_sched[k]['chosen_anchor'] = t1_sched[k]['chosen']
    t1_pc = aggregate(ops, t1_sched)

    # Tier 2: shift + reflect + round-robin
    from rotation_sim_tier2 import rotation_schedule_tier2 as t2_rr
    t2_sched = t2_rr({op['id']: op['activity'] for op in ops})
    t2_pc = aggregate(ops, t2_sched)

    # Tier 3: aging-aware
    t3_sched = schedule_aging_aware(ops)
    t3_pc = aggregate(ops, t3_sched)

    # Comparison
    print('\n=== Baseline vs Tier 1 vs Tier 2 vs Tier 3 ===')
    print(f'{"tier":8s}  {"NIU spread":>12s}  {"NIU worst":>12s}  {"SRAM spread":>12s}  {"SRAM worst":>12s}')
    for name, pc in [('base', base_pc), ('T1 shift', t1_pc),
                     ('T2 refl', t2_pc), ('T3 aware', t3_pc)]:
        s_niu, w_niu = spread(pc, 'niu'), worst_core(pc, 'niu') / 1e6
        s_sr,  w_sr  = spread(pc, 'sram'), worst_core(pc, 'sram') / 1e6
        print(f'{name:8s}  {s_niu:9.2f}×  {w_niu:9.2f} MB  {s_sr:9.2f}×  {w_sr:9.2f} MB')

    # Improvements over baseline
    print('\n=== Improvements over baseline ===')
    b_niu = worst_core(base_pc, 'niu')
    b_sram = worst_core(base_pc, 'sram')
    for name, pc in [('T1', t1_pc), ('T2', t2_pc), ('T3', t3_pc)]:
        w_niu = worst_core(pc, 'niu')
        w_sr  = worst_core(pc, 'sram')
        pct_niu = 100 * (1 - w_niu / b_niu)
        pct_sr  = 100 * (1 - w_sr  / b_sram)
        # HCI lifetime = (u_new/u_old)^-6, NBTI = (u_new/u_old)^-1
        r_niu = w_niu / b_niu
        r_sr  = w_sr  / b_sram
        life_hci_niu = r_niu ** -6
        life_hci_sr  = r_sr  ** -6
        life_nbti_niu = r_niu ** -1
        life_nbti_sr  = r_sr  ** -1
        print(f'  {name}: NIU  worst-core {pct_niu:5.1f}% ↓  → HCI life {life_hci_niu:.1f}×  NBTI life {life_nbti_niu:.1f}×')
        print(f'      SRAM worst-core {pct_sr:5.1f}% ↓  → HCI life {life_hci_sr:.1f}×  NBTI life {life_nbti_sr:.1f}×')

    # Write CSV
    df = pd.concat([to_dataframe(base_pc, 'baseline'),
                    to_dataframe(t1_pc, 'tier1'),
                    to_dataframe(t2_pc, 'tier2'),
                    to_dataframe(t3_pc, 'tier3')], ignore_index=True)
    out = OUT_DIR / 'rotation_tier3_per_core.csv'
    df.to_csv(out, index=False)
    print(f'\nWrote {out}')


if __name__ == '__main__':
    main()

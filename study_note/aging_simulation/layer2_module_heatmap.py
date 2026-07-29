"""Layer 2 per-module heatmap — reveals per-op sharding pattern (rectangular vs L-shape).

Layer 2 (tracy verified):
  - module1 : height_sharding=True → forced HEIGHT_SHARDED
  - module2-4 : height_sharding=None (auto) → auto-picker also picked HEIGHT_SHARDED

All 4 modules are HEIGHT-SHARDED (not block as originally assumed).
Height-sharded factory forces bbox rectangle with noop cells for L-shape.
Aggregating all 29 ops per stage muddles the pattern; per-module split
reveals the actual ~13-core L-shape used by auto-picker.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

REPO = Path('C:/NPU_Challenge')
DEVICE_CSV = REPO / 'study_note/aging_simulation/stage_split_2/stage_2/logs/profile_log_device.csv'
OUT_DIR    = REPO / 'study_note/aging_simulation/plots'
OUT_DIR.mkdir(parents=True, exist_ok=True)

GRID_W, GRID_H = 6, 5
COMPUTE_Y = {0, 1, 2, 4}
COMPUTE_X = set(range(5))
DRAM_Y = 3
DISPATCH_X = 5
TRISC_TYPES = {'TRISC_0', 'TRISC_1', 'TRISC_2'}

ROLE_COLORS = {'compute': '#dbe9f6', 'dram': '#fde2c4', 'dispatch': '#e3e3e3'}
ROLE_BORDERS = {'compute': '#2e6bab', 'dram': '#c26a17', 'dispatch': '#666666'}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 11,
    "axes.linewidth": 0.8, "axes.edgecolor": "#333",
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def classify(x, y):
    if y == DRAM_Y: return 'dram'
    if x == DISPATCH_X: return 'dispatch'
    if x in COMPUTE_X and y in COMPUTE_Y: return 'compute'
    return 'dispatch'


def load_layer2_ops():
    """Return {rhid: mat} — per-op TRISC active cycles matrix."""
    df = pd.read_csv(DEVICE_CSV, skiprows=1)
    df.columns = [c.strip() for c in df.columns]
    df = df[df['RISC processor type'].isin(TRISC_TYPES)]

    starts = df[df['type'] == 'ZONE_START'][
        ['core_x', 'core_y', 'RISC processor type', 'timer_id', 'run host ID', 'time[cycles since reset]']
    ].rename(columns={'time[cycles since reset]': 't_start'})
    ends = df[df['type'] == 'ZONE_END'][
        ['core_x', 'core_y', 'RISC processor type', 'timer_id', 'run host ID', 'time[cycles since reset]']
    ].rename(columns={'time[cycles since reset]': 't_end'})

    key = ['core_x', 'core_y', 'RISC processor type', 'timer_id', 'run host ID']
    paired = starts.merge(ends, on=key, how='inner')
    paired['duration'] = paired['t_end'] - paired['t_start']
    paired = paired[paired['duration'] > 0]

    ops = {}
    for rhid, grp in paired.groupby('run host ID'):
        mat = np.zeros((GRID_H, GRID_W))
        for (x, y), sub in grp.groupby(['core_x', 'core_y']):
            if 0 <= x < GRID_W and 0 <= y < GRID_H:
                mat[y, x] = sub['duration'].sum() / 1e6
        ops[int(rhid)] = mat
    return ops


def group_by_module(op_ids, num_modules=4):
    """Split ordered op ids into num_modules roughly-equal groups."""
    op_ids = sorted(op_ids)
    n = len(op_ids)
    boundaries = [int(n * i / num_modules) for i in range(num_modules + 1)]
    return [op_ids[boundaries[i]:boundaries[i+1]] for i in range(num_modules)]


def sum_ops(ops_dict, op_ids):
    total = np.zeros((GRID_H, GRID_W))
    for oid in op_ids:
        total += ops_dict[oid]
    return total


def draw_role_overlay(ax):
    for y in range(GRID_H):
        for x in range(GRID_W):
            role = classify(x, y)
            ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1,
                                   facecolor=ROLE_COLORS[role], edgecolor=ROLE_BORDERS[role],
                                   linewidth=1.0, zorder=0))
    ax.add_patch(Rectangle((-0.5, DRAM_Y - 0.5), GRID_W, 1,
                           facecolor='none', edgecolor='#c26a17',
                           linewidth=2.0, linestyle='--', zorder=1))


def draw_heatmap_panel(ax, mat, title, cmap='YlOrRd'):
    vmax = float(mat.max()) if mat.max() > 0 else 1.0
    draw_role_overlay(ax)
    for y in range(GRID_H):
        for x in range(GRID_W):
            v = mat[y, x]
            if v > 0:
                shade = min(v / vmax, 1.0)
                color = plt.get_cmap(cmap)(shade)
                ax.add_patch(Rectangle((x - 0.4, y - 0.4), 0.8, 0.8,
                                       facecolor=color, edgecolor='none', alpha=0.9, zorder=2))
            text_color = 'white' if v > 0.5 * vmax else '#111'
            label = f'{v:.2f}' if v >= 0.01 else '·'
            ax.text(x, y, label, ha='center', va='center',
                    color=text_color, fontsize=7.5, zorder=3)
    ax.set_xlim(-0.6, GRID_W - 0.4); ax.set_ylim(GRID_H - 0.4, -0.6)
    ax.set_aspect('equal'); ax.set_xticks(range(GRID_W)); ax.set_yticks(range(GRID_H))
    ax.set_xlabel('sx', fontsize=9); ax.set_ylabel('sy', fontsize=9)
    ax.set_title(title, fontsize=10, pad=4)


def main():
    ops = load_layer2_ops()
    op_ids = sorted(ops.keys())
    modules = group_by_module(op_ids, num_modules=4)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8))
    plt.subplots_adjust(hspace=0.42, wspace=0.20, top=0.90, bottom=0.10)

    module_desc = [
        'Module 1  [height-sharded, forced]',
        'Module 2  [height-sharded, auto-picker chose]',
        'Module 3  [height-sharded, auto-picker chose]',
        'Module 4  [height-sharded, auto-picker chose]',
    ]
    for i, (ax, m_ops, desc) in enumerate(zip(axes.flat, modules, module_desc)):
        mat = sum_ops(ops, m_ops)
        total = mat.sum()
        draw_heatmap_panel(ax, mat,
            f'{desc}\n{len(m_ops)} ops, total {total:.2f} M cycles')

    # Legend
    legend_items = [
        Patch(facecolor=ROLE_COLORS['compute'], edgecolor=ROLE_BORDERS['compute'],
              linewidth=1.2, label='Compute (20 workers)'),
        Patch(facecolor=ROLE_COLORS['dram'], edgecolor=ROLE_BORDERS['dram'],
              linewidth=1.2, label='DRAM row (6 endpoints)'),
        Patch(facecolor=ROLE_COLORS['dispatch'], edgecolor=ROLE_BORDERS['dispatch'],
              linewidth=1.2, label='Dispatch / reserved'),
    ]
    fig.legend(handles=legend_items, loc='lower center',
               bbox_to_anchor=(0.5, 0.02), ncol=3, fontsize=9, frameon=False)

    fig.suptitle('Layer 2 per-module TRISC activity — reveals sharding shape per module\n'
                 f'(stage_2, 29 ops split into 4 modules by op sequence)',
                 fontsize=12, y=0.98)

    out = OUT_DIR / 'layer2_module_heatmaps.png'
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f'Wrote {out}')
    print(f'Module op counts: {[len(m) for m in modules]}')


if __name__ == '__main__':
    main()

"""Per-stage NoC activity heatmaps — annotated style (matches annotated_heatmap.py).

Reveals mcast presence/absence per layer via role-annotated heatmaps.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

REPO = Path('C:/NPU_Challenge')
MATRIX_CSV = REPO / 'study_note/aging_simulation/NoC_analysis/outputs/matrix.csv'
OUT_DIR    = REPO / 'study_note/aging_simulation/plots'
OUT_DIR.mkdir(parents=True, exist_ok=True)

GRID_W, GRID_H = 6, 5

# Roles derived from blackhole_140_arch.yaml
COMPUTE_Y = {0, 1, 2, 4}
COMPUTE_X = set(range(5))
DRAM_Y = 3
DISPATCH_X = 5

ROLE_COLORS = {
    'compute':  '#dbe9f6',
    'dram':     '#fde2c4',
    'dispatch': '#e3e3e3',
}
ROLE_BORDERS = {
    'compute':  '#2e6bab',
    'dram':     '#c26a17',
    'dispatch': '#666666',
}

STAGE_NAMES = {
    0: 'Stem (fold + conv1 + maxpool)',
    1: 'Layer 1',
    2: 'Layer 2',
    3: 'Layer 3',
    4: 'Layer 4 + Head',
}
STAGE_SHARDING = {0: 'height', 1: 'height', 2: 'height', 3: 'block', 4: 'block+width'}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 11,
    "axes.linewidth": 0.8, "axes.edgecolor": "#333",
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def classify(x, y):
    if y == DRAM_Y:
        return 'dram'
    if x == DISPATCH_X:
        return 'dispatch'
    if x in COMPUTE_X and y in COMPUTE_Y:
        return 'compute'
    return 'dispatch'


def load_stage(stage):
    df = pd.read_csv(MATRIX_CSV)
    df = df[(df['is_boundary'] == 0) & (df['stage'] == stage)]
    inj = np.zeros((GRID_H, GRID_W))
    con = np.zeros((GRID_H, GRID_W))
    for (x, y), grp in df.groupby(['sx', 'sy']):
        if 0 <= x < GRID_W and 0 <= y < GRID_H:
            inj[y, x] = grp['injected_bytes'].sum() / 1e6
            con[y, x] = grp['consumed_bytes'].sum() / 1e6
    return inj, con


def draw_role_overlay(ax):
    """Draw role tint + border + DRAM row emphasis."""
    for y in range(GRID_H):
        for x in range(GRID_W):
            role = classify(x, y)
            ax.add_patch(Rectangle(
                (x - 0.5, y - 0.5), 1, 1,
                facecolor=ROLE_COLORS[role],
                edgecolor=ROLE_BORDERS[role],
                linewidth=1.0, zorder=0,
            ))
    # DRAM row emphasis
    ax.add_patch(Rectangle(
        (-0.5, DRAM_Y - 0.5), GRID_W, 1,
        facecolor='none', edgecolor='#c26a17',
        linewidth=2.0, linestyle='--', zorder=1,
    ))


def draw_heatmap_panel(ax, mat, title, cmap='YlOrRd'):
    vmax = float(mat.max()) if mat.max() > 0 else 1.0
    draw_role_overlay(ax)

    for y in range(GRID_H):
        for x in range(GRID_W):
            v = mat[y, x]
            if v > 0:
                shade = min(v / vmax, 1.0)
                color = plt.get_cmap(cmap)(shade)
                ax.add_patch(Rectangle(
                    (x - 0.4, y - 0.4), 0.8, 0.8,
                    facecolor=color, edgecolor='none', alpha=0.9, zorder=2,
                ))
            text_color = 'white' if v > 0.5 * vmax else '#111'
            label = f'{v:.2f}' if v >= 0.01 else ('·' if v < 0.001 else f'{v:.2f}')
            ax.text(x, y, label, ha='center', va='center',
                    color=text_color, fontsize=7.5, zorder=3)

    ax.set_xlim(-0.6, GRID_W - 0.4)
    ax.set_ylim(GRID_H - 0.4, -0.6)   # (0,0) top-left
    ax.set_aspect('equal')
    ax.set_xticks(range(GRID_W))
    ax.set_yticks(range(GRID_H))
    ax.set_xlabel('sx', fontsize=9)
    ax.set_ylabel('sy', fontsize=9)
    ax.set_title(title, fontsize=10, pad=4)


def main():
    fig, axes = plt.subplots(5, 2, figsize=(11.5, 5 * 3.6 + 2.0))
    plt.subplots_adjust(hspace=1.05, wspace=0.20, top=0.925, bottom=0.05)

    for stage in range(5):
        inj, con = load_stage(stage)
        total_inj = inj.sum()
        total_con = con.sum()

        draw_heatmap_panel(axes[stage, 0], inj,
                           f'Injected [MB]  —  total {total_inj:.2f}')
        draw_heatmap_panel(axes[stage, 1], con,
                           f'Consumed [MB]  —  total {total_con:.2f}')

        # Row banner
        row_title = f"Stage {stage}  —  {STAGE_NAMES[stage]}   [{STAGE_SHARDING[stage]}-sharded]"
        pos = axes[stage, 0].get_position()
        fig.text(0.5, pos.y1 + 0.028, row_title,
                 ha='center', fontsize=12, fontweight='bold', color='#222')

    # Global legend at bottom
    legend_items = [
        Patch(facecolor=ROLE_COLORS['compute'], edgecolor=ROLE_BORDERS['compute'],
              linewidth=1.2, label='Compute (20 workers)'),
        Patch(facecolor=ROLE_COLORS['dram'], edgecolor=ROLE_BORDERS['dram'],
              linewidth=1.2, label='DRAM row (6 endpoints)'),
        Patch(facecolor=ROLE_COLORS['dispatch'], edgecolor=ROLE_BORDERS['dispatch'],
              linewidth=1.2, label='Dispatch / reserved'),
    ]
    fig.legend(handles=legend_items, loc='lower center',
               bbox_to_anchor=(0.5, 0.005), ncol=3, fontsize=9, frameon=False)

    fig.suptitle('Per-stage NoC activity — mcast pattern reveals sharding scheme',
                 fontsize=13.5, y=0.985)

    out = OUT_DIR / 'per_stage_heatmaps.png'
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()

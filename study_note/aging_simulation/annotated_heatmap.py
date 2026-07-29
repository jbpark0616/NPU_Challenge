"""Role-annotated per-core NoC heatmap for Eagle-N (Blackhole 140 arch).

Overlays yaml-derived role labels onto the 6x5 raw NoC trace grid:
  - Compute (blue border) : y in {0,1,2,4}, x in {0..4} = 20 workers
  - DRAM    (orange fill) : y == 3 = memory row (6 endpoints)
  - Dispatch (gray fill)  : x == 5 (except y=3) = 4 reserved endpoints

Physical adjacency note: y=3 DRAM row is *physically between* compute rows
y=2 and y=4, so those two compute rows are 2 hops apart on the NoC.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

REPO = Path('C:/NPU_Challenge')
NOC_CSV = REPO / 'bos_model_copy/resnet50_jbpark/analysis/outputs/per_core.csv'
OUT_DIR = REPO / 'study_note/aging_simulation/plots'
OUT_DIR.mkdir(parents=True, exist_ok=True)

GRID_W, GRID_H = 6, 5

# Roles derived from blackhole_140_arch.yaml
COMPUTE_Y = {0, 1, 2, 4}
COMPUTE_X = set(range(5))
DRAM_Y = 3
DISPATCH_X = 5

ROLE_COLORS = {
    'compute':  '#dbe9f6',   # light blue
    'dram':     '#fde2c4',   # light orange
    'dispatch': '#e3e3e3',   # gray
}
ROLE_BORDERS = {
    'compute':  '#2e6bab',
    'dram':     '#c26a17',
    'dispatch': '#666666',
}

def classify(x, y):
    if y == DRAM_Y:
        return 'dram'
    if x == DISPATCH_X:
        return 'dispatch'
    if x in COMPUTE_X and y in COMPUTE_Y:
        return 'compute'
    return 'dispatch'   # fallback


def load_grid():
    df = pd.read_csv(NOC_CSV)
    inj = np.zeros((GRID_H, GRID_W))
    con = np.zeros((GRID_H, GRID_W))
    for _, row in df.iterrows():
        x, y = int(row['sx']), int(row['sy'])
        if 0 <= x < GRID_W and 0 <= y < GRID_H:
            inj[y, x] = row['injected_bytes'] / 1e6   # MB
            con[y, x] = row['consumed_bytes'] / 1e6
    return inj, con


def draw_role_overlay(ax):
    """Draw colored rectangle behind each cell to mark role, plus role border."""
    for y in range(GRID_H):
        for x in range(GRID_W):
            role = classify(x, y)
            # Solid role tint underneath
            rect = Rectangle(
                (x - 0.5, y - 0.5), 1, 1,
                facecolor=ROLE_COLORS[role],
                edgecolor=ROLE_BORDERS[role],
                linewidth=1.2, zorder=0,
            )
            ax.add_patch(rect)
    # Explicit DRAM row band emphasis (physical adjacency reminder)
    ax.add_patch(Rectangle(
        (-0.5, DRAM_Y - 0.5), GRID_W, 1,
        facecolor='none', edgecolor='#c26a17',
        linewidth=2.5, linestyle='--', zorder=1,
    ))


def draw_heatmap_panel(ax, mat, title, cmap='YlOrRd', vmax=None):
    if vmax is None:
        vmax = float(mat.max())

    draw_role_overlay(ax)

    # Data overlay: alpha-blended on top of role tint
    for y in range(GRID_H):
        for x in range(GRID_W):
            v = mat[y, x]
            if v > 0:
                shade = min(v / vmax, 1.0)
                color = plt.get_cmap(cmap)(shade)
                # Draw semi-transparent shaded square
                ax.add_patch(Rectangle(
                    (x - 0.4, y - 0.4), 0.8, 0.8,
                    facecolor=color, edgecolor='none', alpha=0.85, zorder=2,
                ))
            text_color = 'white' if v > 0.5 * vmax else 'black'
            ax.text(x, y, f'{v:.2f}' if v > 0 else '·',
                    ha='center', va='center',
                    color=text_color, fontsize=9, zorder=3)

    ax.set_xlim(-0.6, GRID_W - 0.4)
    ax.set_ylim(GRID_H - 0.4, -0.6)   # (0,0) top-left convention
    ax.set_aspect('equal')
    ax.set_xticks(range(GRID_W))
    ax.set_yticks(range(GRID_H))
    ax.set_xlabel('sx (NoC col)')
    ax.set_ylabel('sy (NoC row)')
    ax.set_title(title)

    # Legend
    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor=ROLE_COLORS['compute'], edgecolor=ROLE_BORDERS['compute'],
              linewidth=1.2, label='Compute (20 workers)'),
        Patch(facecolor=ROLE_COLORS['dram'], edgecolor=ROLE_BORDERS['dram'],
              linewidth=1.2, label='DRAM row (6 endpoints)'),
        Patch(facecolor=ROLE_COLORS['dispatch'], edgecolor=ROLE_BORDERS['dispatch'],
              linewidth=1.2, label='Dispatch / reserved'),
    ]
    ax.legend(handles=legend_items, loc='upper center',
              bbox_to_anchor=(0.5, -0.10), ncol=3, fontsize=8, frameon=False)


def main():
    inj, con = load_grid()

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    draw_heatmap_panel(axes[0], inj, 'Injected NoC bytes [MB]',
                       vmax=inj.max())
    draw_heatmap_panel(axes[1], con, 'Consumed NoC bytes [MB]',
                       vmax=con.max())

    fig.suptitle('Eagle-N NoC — 20 compute + 6 DRAM + 4 dispatch  |  ResNet50 batch=1, real ops',
                 fontsize=12, y=0.98)

    fig.text(0.5, 0.02,
             'Physical layout: y=0,1,2 compute stacked → y=3 DRAM row → y=4 compute. '
             'Rows y=2 and y=4 are 2 NoC hops apart (must cross DRAM row).',
             ha='center', fontsize=9, style='italic', color='#444')

    plt.subplots_adjust(top=0.90, bottom=0.16, wspace=0.15)
    out = OUT_DIR / 'per_core_heatmap_annotated.png'
    plt.savefig(out, dpi=140, bbox_inches='tight')
    plt.close()
    print(f'Wrote {out}')

if __name__ == '__main__':
    main()

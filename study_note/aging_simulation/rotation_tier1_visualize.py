"""Visualize Tier 1 rotation vs baseline — per-core activity heatmaps.

Compares baseline (no rotation) vs Tier 1 (anchor shift + round-robin)
side-by-side for NIU and SRAM axes.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

REPO = Path('C:/NPU_Challenge')
CSV = REPO / 'study_note/aging_simulation/plots/rotation_tier1_per_core.csv'
OUT_DIR = REPO / 'study_note/aging_simulation/plots'

GRID_W = 5   # logical x
GRID_H = 4   # logical y
LOG_TO_PHYS_Y = {0: 0, 1: 1, 2: 2, 3: 4}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 11,
    "axes.linewidth": 0.8, "axes.edgecolor": "#333",
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
})

CMAP = 'YlOrRd'


def load_data():
    df = pd.read_csv(CSV)
    return df


def to_matrix(df, source, col):
    """Return (GRID_H, GRID_W) matrix in logical coords."""
    mat = np.zeros((GRID_H, GRID_W))
    sub = df[df['source'] == source]
    for _, r in sub.iterrows():
        lx, ly = int(r['lx']), int(r['ly'])
        if 0 <= lx < GRID_W and 0 <= ly < GRID_H:
            mat[ly, lx] = r[col] / 1e6   # MB
    return mat


def draw_heatmap(ax, mat, title, vmax_shared=None, show_ylabel=True):
    vmax = vmax_shared if vmax_shared else mat.max()
    im = ax.imshow(mat, cmap=CMAP, vmin=0, vmax=vmax, aspect='equal')

    # Value overlays
    for (i, j), v in np.ndenumerate(mat):
        color = 'white' if v > 0.55 * vmax else '#111'
        label = f'{v:.2f}' if v > 0.01 else '·'
        ax.text(j, i, label, ha='center', va='center',
                color=color, fontsize=9, fontweight='medium')

    ax.set_xticks(range(GRID_W))
    ax.set_yticks(range(GRID_H))
    ax.set_yticklabels([LOG_TO_PHYS_Y[y] for y in range(GRID_H)])
    ax.set_xlabel('core_x')
    if show_ylabel:
        ax.set_ylabel('core_y (physical)')
    ax.set_title(title, pad=6)

    cb = plt.colorbar(im, ax=ax, fraction=0.038, pad=0.03)
    cb.set_label('MB', rotation=90, labelpad=8)
    cb.ax.tick_params(labelsize=8)
    return im


def main():
    df = load_data()

    # 4-panel: rows = axes (niu, sram), cols = (baseline, rotation)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    plt.subplots_adjust(hspace=0.45, wspace=0.25)

    axis_labels = [
        ('total_noc', 'NIU  (injected + consumed)'),
        ('consumed_bytes', 'SRAM  (consumed only)'),
    ]

    for row, (col_name, axis_title) in enumerate(axis_labels):
        base = to_matrix(df, 'baseline', col_name)
        rot  = to_matrix(df, 'rotation_rr', col_name)
        vmax = max(base.max(), rot.max())

        draw_heatmap(axes[row, 0], base,
                     f'{axis_title}\nBaseline (no rotation)   spread = {base.max()/base[base>0].min():.1f}×',
                     vmax_shared=vmax, show_ylabel=True)
        draw_heatmap(axes[row, 1], rot,
                     f'{axis_title}\nTier 1 rotation (round-robin)   spread = {rot.max()/rot[rot>0].min():.1f}×',
                     vmax_shared=vmax, show_ylabel=False)

    fig.suptitle('Baseline vs Tier 1 rotation (anchor shift + round-robin)\n'
                 '58% of ops use full grid (no rotation possible) — modest improvement',
                 fontsize=12, y=0.995)

    out = OUT_DIR / 'rotation_tier1_comparison.png'
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()

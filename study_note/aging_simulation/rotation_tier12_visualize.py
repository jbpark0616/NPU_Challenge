"""3-way comparison: Baseline vs Tier 1 (shift) vs Tier 2 (shift + reflect)."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO = Path('C:/NPU_Challenge')
CSV = REPO / 'study_note/aging_simulation/plots/rotation_tier2_per_core.csv'
OUT_DIR = REPO / 'study_note/aging_simulation/plots'

GRID_W = 5
GRID_H = 4
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


def to_matrix(df, source, col):
    mat = np.zeros((GRID_H, GRID_W))
    sub = df[df['source'] == source]
    for _, r in sub.iterrows():
        lx, ly = int(r['lx']), int(r['ly'])
        if 0 <= lx < GRID_W and 0 <= ly < GRID_H:
            mat[ly, lx] = r[col] / 1e6
    return mat


def draw(ax, mat, title, vmax):
    im = ax.imshow(mat, cmap=CMAP, vmin=0, vmax=vmax, aspect='equal')
    for (i, j), v in np.ndenumerate(mat):
        color = 'white' if v > 0.55 * vmax else '#111'
        label = f'{v:.2f}' if v > 0.01 else '·'
        ax.text(j, i, label, ha='center', va='center',
                color=color, fontsize=8.5, fontweight='medium')
    ax.set_xticks(range(GRID_W))
    ax.set_yticks(range(GRID_H))
    ax.set_yticklabels([LOG_TO_PHYS_Y[y] for y in range(GRID_H)])
    ax.set_xlabel('core_x')
    ax.set_ylabel('core_y (phys)')
    ax.set_title(title, pad=6, fontsize=10)
    cb = plt.colorbar(im, ax=ax, fraction=0.038, pad=0.03)
    cb.set_label('MB', rotation=90, labelpad=8)
    cb.ax.tick_params(labelsize=8)


def main():
    df = pd.read_csv(CSV)

    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    plt.subplots_adjust(hspace=0.42, wspace=0.28)

    for row, (col_name, axis_title) in enumerate([
        ('total_noc', 'NIU  (injected + consumed)'),
        ('consumed_bytes', 'SRAM  (consumed only)'),
    ]):
        mats = [to_matrix(df, src, col_name)
                for src in ['baseline', 'tier1', 'tier2']]
        vmax = max(m.max() for m in mats)

        for col, (mat, tag) in enumerate(zip(mats, ['Baseline', 'Tier 1: shift', 'Tier 2: + reflect'])):
            valid = mat[mat > 0]
            spread = mat.max() / valid.min() if len(valid) > 0 else 0
            draw(axes[row, col], mat,
                 f'{axis_title}\n{tag}   spread = {spread:.1f}×',
                 vmax=vmax)

    fig.suptitle('Rotation progression: baseline → shift → shift + reflection\n'
                 'Tier 2 = 8 configs for block-sharded 4×4 ops (14 ops, 16% of workload but ~70% of traffic)',
                 fontsize=12, y=0.995)

    out = OUT_DIR / 'rotation_tier12_comparison.png'
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()

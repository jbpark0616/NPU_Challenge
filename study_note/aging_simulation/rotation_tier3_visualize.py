"""4-way comparison heatmap: Baseline / Tier1 / Tier2 / Tier3."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO = Path('C:/NPU_Challenge')
CSV = REPO / 'study_note/aging_simulation/plots/rotation_tier3_per_core.csv'
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
                color=color, fontsize=8, fontweight='medium')
    ax.set_xticks(range(GRID_W))
    ax.set_yticks(range(GRID_H))
    ax.set_yticklabels([LOG_TO_PHYS_Y[y] for y in range(GRID_H)])
    ax.set_xlabel('core_x', fontsize=9)
    ax.set_ylabel('core_y (phys)', fontsize=9)
    ax.set_title(title, pad=6, fontsize=9.5)
    cb = plt.colorbar(im, ax=ax, fraction=0.038, pad=0.03)
    cb.set_label('MB', rotation=90, labelpad=8, fontsize=8)
    cb.ax.tick_params(labelsize=7)


def main():
    df = pd.read_csv(CSV)

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    plt.subplots_adjust(hspace=0.42, wspace=0.32)

    tiers = [
        ('baseline', 'Baseline (no rotation)'),
        ('tier1',    'Tier 1: shift + RR'),
        ('tier2',    'Tier 2: + reflect + RR'),
        ('tier3',    'Tier 3: aging-aware\n(Brandalero DAC\'20 style)'),
    ]

    for row, (col_name, axis_title) in enumerate([
        ('total_noc', 'NIU  (injected + consumed)'),
        ('consumed_bytes', 'SRAM  (consumed only)'),
    ]):
        mats = [to_matrix(df, src, col_name) for src, _ in tiers]
        vmax = max(m.max() for m in mats)

        for col, ((src, tag), mat) in enumerate(zip(tiers, mats)):
            valid = mat[mat > 0]
            spread = mat.max() / valid.min() if len(valid) > 0 else 0
            worst = mat.max()
            title = f'{axis_title}\n{tag}\nworst = {worst:.2f} MB   spread = {spread:.1f}×'
            draw(axes[row, col], mat, title, vmax=vmax)

    fig.suptitle('Rotation strategy progression — baseline → shift → reflect → aging-aware\n'
                 'Tier 3 mirrors Brandalero DAC\'20 utilization-aware allocation, adapted to programmable NPU',
                 fontsize=12, y=1.00)

    out = OUT_DIR / 'rotation_tier3_comparison.png'
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()

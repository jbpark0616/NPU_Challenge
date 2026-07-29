"""Per-stage sharding distribution — PPT-ready.

Data: grep counts of memory_layout tags from tracy_ops_data.csv per stage.
Message: HEIGHT/BLOCK are the mainstream targets; WIDTH is a small tail in
Stage 4 only (Head / Matmul-like), so it is out of our rotation scope.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT_DIR = Path('C:/NPU_Challenge/study_note/aging_simulation/plots')

stages = ['Stage 0\n(Stem)',
          'Stage 1\n(Layer 1)',
          'Stage 2\n(Layer 2)',
          'Stage 3\n(Layer 3)',
          'Stage 4\n(Layer 4 + Head)']

data = {
    'HEIGHT_SHARDED': [24, 29, 60,  0,  0],
    'BLOCK_SHARDED':  [ 0,  0,  0, 36, 44],
    'WIDTH_SHARDED':  [ 0,  0,  0,  0, 27],
    'INTERLEAVED':    [ 2, 11, 13, 42, 28],
}

COLORS = {
    'HEIGHT_SHARDED': '#3498db',
    'BLOCK_SHARDED':  '#e67e22',
    'WIDTH_SHARDED':  '#c0392b',
    'INTERLEAVED':    '#bdc3c7',
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 11,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333",
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def main():
    fig, ax = plt.subplots(figsize=(13, 6.2))

    x = np.arange(len(stages))
    n_series = len(data)
    bar_w = 0.19

    for i, (label, values) in enumerate(data.items()):
        offset = (i - (n_series - 1) / 2) * bar_w
        bars = ax.bar(x + offset, values, bar_w,
                      label=label, color=COLORS[label],
                      edgecolor='#333', linewidth=0.7)
        for j, v in enumerate(values):
            if v > 0:
                ax.text(x[j] + offset, v + 1.2, str(v),
                        ha='center', va='bottom',
                        fontsize=9.5, fontweight='bold',
                        color='#333')

    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=11)
    ax.set_ylabel('op count (per stage)', fontsize=11)
    ax.set_title('Per-stage sharding distribution — ResNet50 batch=1 '
                 '(from tracy_ops_data.csv)',
                 fontsize=13, fontweight='bold', pad=12)

    ax.set_ylim(0, 70)
    ax.grid(axis='y', linestyle='--', alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)

    ax.legend(loc='upper left', fontsize=10, frameon=True,
              framealpha=0.9, edgecolor='#ccc')

    # Callout for Stage 4 WIDTH_SHARDED
    ax.annotate(
        'WIDTH_SHARDED only appears\nin Stage 4 (Head / Matmul-like ops).\n'
        'Separate factory: multi_core_conv2d_width_sharded\n'
        '→ out of our rotation scope',
        xy=(4.19, 27), xycoords='data',
        xytext=(2.65, 58), textcoords='data',
        fontsize=9.5, color='#c0392b',
        ha='left', va='center',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#fdedec',
                  edgecolor='#c0392b', linewidth=1.2),
        arrowprops=dict(arrowstyle='->', color='#c0392b',
                        lw=1.4, connectionstyle='arc3,rad=-0.15'))

    # Footer note
    fig.text(0.5, -0.02,
             'HEIGHT (Layer 1-2) and BLOCK (Layer 3-4) share the same '
             'sharded_program_factory.cpp — our rotation targets.  '
             'INTERLEAVED = DRAM tensor layout (not compute sharding).',
             ha='center', fontsize=9.5, color='#555', style='italic')

    out = OUT_DIR / 'sharding_distribution_per_stage.png'
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()

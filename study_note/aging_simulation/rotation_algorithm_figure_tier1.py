"""Paper-style figure showing Tier 1 rotation algorithm.

Illustrates:
  (a) An op shape (4x2 example) with its 'hot corner' at anchor
  (b) All valid anchor positions on the 5x4 compute grid (strict fit)
  (c) Cyclic anchor rotation — 6 successive placements
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow

OUT_DIR = Path('C:/NPU_Challenge/study_note/aging_simulation/plots')

# Grid
GRID_W = 5
GRID_H = 4
LOG_TO_PHYS_Y = {0: 0, 1: 1, 2: 2, 3: 4}

# Example op shape
SHAPE_W = 4
SHAPE_H = 2

# Colors
COL_GRID_BG = '#f5f5f5'
COL_COMPUTE_BORDER = '#7f8c8d'
COL_SHAPE_FILL = '#f39c12'
COL_SHAPE_HOT = '#c0392b'
COL_SHAPE_BORDER = '#d35400'
COL_INACTIVE = '#ecf0f1'

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333",
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def valid_anchors(w, h):
    return [(x, y) for y in range(GRID_H - h + 1)
                   for x in range(GRID_W - w + 1)]


def draw_grid(ax, cell_size=1.0):
    """Draw empty compute grid."""
    for y in range(GRID_H):
        for x in range(GRID_W):
            ax.add_patch(Rectangle(
                (x, GRID_H - 1 - y), cell_size, cell_size,
                facecolor=COL_INACTIVE, edgecolor=COL_COMPUTE_BORDER,
                linewidth=1.0, zorder=1))
    ax.set_xlim(-0.15, GRID_W + 0.15)
    ax.set_ylim(-0.15, GRID_H + 0.15)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_shape_at_anchor(ax, anchor_x, anchor_y, shape_w, shape_h,
                          hot_at='top-left', alpha=0.9,
                          step_num=None, mark_hot=True):
    """Draw the shape's cells at the given anchor. Hot corner highlighted."""
    for dy in range(shape_h):
        for dx in range(shape_w):
            x = anchor_x + dx
            y = anchor_y + dy
            # Flip y for display (matplotlib origin bottom-left)
            display_y = GRID_H - 1 - y

            # Hot corner
            if mark_hot and dx == 0 and dy == 0:
                color = COL_SHAPE_HOT
            else:
                color = COL_SHAPE_FILL

            ax.add_patch(Rectangle(
                (x, display_y), 1, 1,
                facecolor=color, edgecolor=COL_SHAPE_BORDER,
                linewidth=1.5, alpha=alpha, zorder=3))

    if step_num is not None:
        ax.text(anchor_x + shape_w / 2, GRID_H - anchor_y - shape_h / 2,
                str(step_num), ha='center', va='center',
                fontsize=18, fontweight='bold', color='white', zorder=5)


def draw_axes_labels(ax):
    """Add coord labels around grid."""
    for x in range(GRID_W):
        ax.text(x + 0.5, -0.35, str(x), ha='center', va='top',
                fontsize=8, color='#666')
    for y in range(GRID_H):
        ax.text(-0.25, GRID_H - y - 0.5, str(LOG_TO_PHYS_Y[y]),
                ha='right', va='center', fontsize=8, color='#666')
    ax.text(GRID_W / 2, -0.85, 'core_x', ha='center', fontsize=9)
    ax.text(-0.9, GRID_H / 2, 'core_y (phys)', va='center',
            rotation=90, fontsize=9)


def main():
    anchors = valid_anchors(SHAPE_W, SHAPE_H)
    n_anchors = len(anchors)   # 6 for 4x2 on 5x4

    # Layout: shape reference (left), 6 round-robin steps (right in 2x3 grid)
    fig = plt.figure(figsize=(15, 6.5))

    # ============================================================
    # Left: shape reference + legend (spans full height)
    # ============================================================
    ax_ref = fig.add_axes([0.02, 0.10, 0.24, 0.78])
    ax_ref.set_xlim(0, 5.0)
    ax_ref.set_ylim(0, 8)
    ax_ref.set_aspect('equal')
    ax_ref.axis('off')

    # ---- Shape header ----
    ax_ref.text(2.5, 7.5,
                f'Op shape:  {SHAPE_W} x {SHAPE_H}',
                ha='center', fontsize=13, fontweight='bold')

    # ---- Draw shape ----
    y_shape_top = 6.6
    x_shape_left = 0.7
    for dy in range(SHAPE_H):
        for dx in range(SHAPE_W):
            color = COL_SHAPE_HOT if (dx == 0 and dy == 0) else COL_SHAPE_FILL
            ax_ref.add_patch(Rectangle(
                (x_shape_left + dx * 0.7, y_shape_top - dy * 0.7),
                0.7, 0.7,
                facecolor=color, edgecolor=COL_SHAPE_BORDER, linewidth=1.5))

    # ---- Hot corner annotation (arrow) ----
    ax_ref.annotate(
        'Hot corner\n(mcast anchor)',
        xy=(x_shape_left + 0.35, y_shape_top + 0.35),
        xytext=(3.7, 6.9),
        fontsize=9, color=COL_SHAPE_HOT, fontweight='bold',
        ha='left', va='center',
        arrowprops=dict(arrowstyle='->', color=COL_SHAPE_HOT, lw=1.3))

    # ---- Text block ----
    exp_lines = [
        'Valid anchors on 5x4 grid:',
        f'  x in 0..{GRID_W - SHAPE_W},  y in 0..{GRID_H - SHAPE_H}',
        f'  = {GRID_W - SHAPE_W + 1} x {GRID_H - SHAPE_H + 1} = {n_anchors} positions',
        '',
        'Tier 1 policy',
        '  Cyclic anchor rotation:',
        '  sweep the 6 anchors in order.',
        '',
        '  Op #k -> anchor[k mod 6]',
    ]
    y_start = 4.0
    for i, line in enumerate(exp_lines):
        weight = 'bold' if line in ('Valid anchors on 5x4 grid:',
                                     'Tier 1 policy') else 'normal'
        ax_ref.text(0.3, y_start - i * 0.42, line, ha='left', va='top',
                    fontsize=10, color='#333', fontweight=weight)

    # ============================================================
    # Right: 2×3 grid of round-robin sequence
    # ============================================================
    ax_positions = [
        [0.31, 0.53, 0.22, 0.33],   # top-left
        [0.55, 0.53, 0.22, 0.33],   # top-middle
        [0.79, 0.53, 0.22, 0.33],   # top-right
        [0.31, 0.10, 0.22, 0.33],   # bot-left
        [0.55, 0.10, 0.22, 0.33],   # bot-middle
        [0.79, 0.10, 0.22, 0.33],   # bot-right
    ]

    for i, (anchor, pos) in enumerate(zip(anchors, ax_positions)):
        ax = fig.add_axes(pos)
        draw_grid(ax)
        draw_shape_at_anchor(ax, anchor[0], anchor[1], SHAPE_W, SHAPE_H,
                              step_num=i + 1)
        draw_axes_labels(ax)
        ax.set_title(f'Step {i+1}:   anchor = ({anchor[0]}, {anchor[1]})',
                     fontsize=10.5, pad=6)

    # Suptitle
    fig.text(0.63, 0.98,
             'Tier 1 rotation: cyclic anchor rotation',
             ha='center', fontsize=14, fontweight='bold')
    fig.text(0.63, 0.94,
             f'Each successive op executes at the next anchor in cycle. '
             f'Shape preserved, only pivot shifts.',
             ha='center', fontsize=10, color='#555', style='italic')

    out = OUT_DIR / 'algorithm_tier1_illustration.png'
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()

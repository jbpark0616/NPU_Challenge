"""Paper-style figure showing Tier 2 rotation algorithm.

Tier 2 = Tier 1 (anchor shift) + Reflection (block-sharded only).
For 4x4 shape on 5x4 grid:
  - 2 anchors x 4 reflections = 8 configs

The reflection modes swap which edge is the mcast sender:
  - TL (default): weight sender = top row,    activation sender = left col
  - TR:            weight sender = top row,    activation sender = right col
  - BL:            weight sender = bottom row, activation sender = left col
  - BR:            weight sender = bottom row, activation sender = right col

Hot corner (A+W double duty) moves to the intersection of sender edges.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow

OUT_DIR = Path('C:/NPU_Challenge/study_note/aging_simulation/plots')

GRID_W = 5
GRID_H = 4
LOG_TO_PHYS_Y = {0: 0, 1: 1, 2: 2, 3: 4}

SHAPE_W = 4   # block-sharded CORE_SET_16
SHAPE_H = 4

COL_INACTIVE = '#ecf0f1'
COL_COMPUTE_BORDER = '#7f8c8d'
COL_SHAPE_FILL = '#f39c12'
COL_SHAPE_HOT = '#c0392b'
COL_SHAPE_BORDER = '#d35400'
COL_WEIGHT_SENDER = '#e67e22'
COL_ACT_SENDER = '#f5b041'
COL_INTERIOR = '#f9e79f'

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


# Reflection specs — which corner is hot, which edge is W sender, which is A sender
# For each mode: (hot_corner, weight_sender_edge, activation_sender_edge)
REFLECTIONS = {
    'TL': {'hot': (0, 0),           'w_edge': 'top',    'a_edge': 'left'},
    'TR': {'hot': (SHAPE_W - 1, 0), 'w_edge': 'top',    'a_edge': 'right'},
    'BL': {'hot': (0, SHAPE_H - 1), 'w_edge': 'bottom', 'a_edge': 'left'},
    'BR': {'hot': (SHAPE_W - 1, SHAPE_H - 1),
                                    'w_edge': 'bottom', 'a_edge': 'right'},
}


def cell_color_for_reflected_shape(dx, dy, W, H, mode):
    """Given local (dx, dy), return the cell color based on reflection mode."""
    info = REFLECTIONS[mode]
    hx, hy = info['hot']

    # Hot corner
    if (dx, dy) == (hx, hy):
        return COL_SHAPE_HOT

    # Weight sender row (top or bottom)
    is_w_sender_row = (info['w_edge'] == 'top'    and dy == 0) or \
                      (info['w_edge'] == 'bottom' and dy == H - 1)
    # Activation sender col (left or right)
    is_a_sender_col = (info['a_edge'] == 'left'  and dx == 0) or \
                      (info['a_edge'] == 'right' and dx == W - 1)

    if is_w_sender_row and is_a_sender_col:
        return COL_SHAPE_HOT   # intersection (also hot)
    if is_w_sender_row:
        return COL_WEIGHT_SENDER
    if is_a_sender_col:
        return COL_ACT_SENDER
    return COL_INTERIOR


def draw_shape_reflected(ax, x_off, y_off, W, H, mode, cell_size=1.0):
    for dy in range(H):
        for dx in range(W):
            color = cell_color_for_reflected_shape(dx, dy, W, H, mode)
            ax.add_patch(Rectangle(
                (x_off + dx * cell_size, y_off + (H - 1 - dy) * cell_size),
                cell_size, cell_size,
                facecolor=color, edgecolor=COL_SHAPE_BORDER,
                linewidth=1.2, zorder=3))


def draw_grid(ax):
    for y in range(GRID_H):
        for x in range(GRID_W):
            ax.add_patch(Rectangle(
                (x, GRID_H - 1 - y), 1, 1,
                facecolor=COL_INACTIVE, edgecolor=COL_COMPUTE_BORDER,
                linewidth=1.0, zorder=1))
    ax.set_xlim(-0.15, GRID_W + 0.15)
    ax.set_ylim(-0.15, GRID_H + 0.15)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_shape_on_grid(ax, anchor_x, anchor_y, W, H, mode):
    """Place reflected shape at anchor on the grid axis."""
    for dy in range(H):
        for dx in range(W):
            color = cell_color_for_reflected_shape(dx, dy, W, H, mode)
            gx = anchor_x + dx
            gy = anchor_y + dy
            display_y = GRID_H - 1 - gy
            ax.add_patch(Rectangle(
                (gx, display_y), 1, 1,
                facecolor=color, edgecolor=COL_SHAPE_BORDER,
                linewidth=1.5, alpha=0.92, zorder=3))


def draw_axes_labels(ax):
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
    fig = plt.figure(figsize=(18, 9))

    # Suptitle
    fig.text(0.5, 0.985,
             'Tier 2 rotation: anchor shift + reflection (block-sharded)',
             ha='center', fontsize=15, fontweight='bold')
    fig.text(0.5, 0.955,
             '4x4 shape on 5x4 grid: 2 anchors x 4 reflections = 8 configs. '
             'Reflection swaps which edge is the mcast sender (hot corner moves).',
             ha='center', fontsize=10.5, color='#555', style='italic')

    # =====================================================
    # TOP ROW: 4 reflection modes shown as shape-only diagrams
    # =====================================================
    modes = ['TL', 'TR', 'BL', 'BR']
    mode_descriptions = {
        'TL': ('TL  (default)',    'W sender: top,    A sender: left'),
        'TR': ('TR  (flip H)',     'W sender: top,    A sender: right'),
        'BL': ('BL  (flip V)',     'W sender: bottom, A sender: left'),
        'BR': ('BR  (flip both)',  'W sender: bottom, A sender: right'),
    }

    for i, mode in enumerate(modes):
        # Position axes for shape-only view
        left  = 0.06 + i * 0.235
        ax = fig.add_axes([left, 0.64, 0.15, 0.22])
        ax.set_xlim(-0.3, SHAPE_W + 0.3)
        ax.set_ylim(-0.3, SHAPE_H + 0.3)
        ax.set_aspect('equal')
        ax.axis('off')

        draw_shape_reflected(ax, 0, 0, SHAPE_W, SHAPE_H, mode)

        title, subtitle = mode_descriptions[mode]
        ax.text(SHAPE_W / 2, SHAPE_H + 0.8, title,
                ha='center', fontsize=12, fontweight='bold')
        ax.text(SHAPE_W / 2, SHAPE_H + 0.25, subtitle,
                ha='center', fontsize=9, color='#555')

    # =====================================================
    # BOTTOM ROW: 8 configs = 2 anchors x 4 reflections
    # =====================================================
    # Two anchors for 4x4: (0,0) and (1,0)
    anchors = [(0, 0), (1, 0)]

    # 8 sub-axes in 2 rows x 4 cols
    # Row 0 (anchor 0,0): bottom=0.34, height=0.20  -> covers 0.34..0.54
    # Row 1 (anchor 1,0): bottom=0.08, height=0.20  -> covers 0.08..0.28
    row_bottoms = [0.34, 0.08]
    for row, anchor in enumerate(anchors):
        for col, mode in enumerate(modes):
            left = 0.06 + col * 0.235
            bottom = row_bottoms[row]
            ax = fig.add_axes([left, bottom, 0.15, 0.20])
            draw_grid(ax)
            draw_shape_on_grid(ax, anchor[0], anchor[1], SHAPE_W, SHAPE_H, mode)
            draw_axes_labels(ax)

            step_num = row * 4 + col + 1
            ax.set_title(f'Step {step_num}:  anchor=({anchor[0]},{anchor[1]}) + {mode}',
                         fontsize=9, pad=6)

    # =====================================================
    # Legend (bottom of figure)
    # =====================================================
    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor=COL_SHAPE_HOT, edgecolor=COL_SHAPE_BORDER,
              label='Hot corner (W+A sender intersection)'),
        Patch(facecolor=COL_WEIGHT_SENDER, edgecolor=COL_SHAPE_BORDER,
              label='Weight mcast sender (row)'),
        Patch(facecolor=COL_ACT_SENDER, edgecolor=COL_SHAPE_BORDER,
              label='Activation mcast sender (col)'),
        Patch(facecolor=COL_INTERIOR, edgecolor=COL_SHAPE_BORDER,
              label='Interior (receiver only)'),
    ]
    fig.legend(handles=legend_items, loc='lower center',
               bbox_to_anchor=(0.5, 0.005), ncol=4, fontsize=9.5,
               frameon=False)

    # Separator line between shape row and grid row
    fig.text(0.5, 0.605,
             'v v v  Apply each reflection at 2 valid anchors  v v v',
             ha='center', fontsize=10, color='#888', fontweight='bold')

    out = OUT_DIR / 'algorithm_tier2_illustration.png'
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()

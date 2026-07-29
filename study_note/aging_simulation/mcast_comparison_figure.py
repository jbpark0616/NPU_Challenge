"""Mcast structural comparison figure — Height vs Block vs Width (PPT-ready).

For each sharding type, show:
  - Sender cells (mcast source, high NIU/BW load)
  - Receiver cells
  - Hot corner (Block only: row & col sender intersection)
  - Not used (blank) cells

Also annotates the key structural properties (mcast count, rotation freedom).
"""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch

OUT_DIR = Path('C:/NPU_Challenge/study_note/aging_simulation/plots')

GRID_W = 5
GRID_H = 4
LOG_TO_PHYS_Y = {0: 0, 1: 1, 2: 2, 3: 4}

COL_INACTIVE = '#ecf0f1'
COL_COMPUTE_BORDER = '#7f8c8d'
COL_SHAPE_BORDER = '#d35400'
COL_HOT = '#c0392b'
COL_W_SENDER = '#e67e22'
COL_A_SENDER = '#f5b041'
COL_INTERIOR = '#f9e79f'
COL_SINGLE_SENDER = '#c0392b'
COL_RECEIVER = '#f9e79f'
COL_BROADCAST_SRC = '#8e44ad'
COL_BROADCAST_DST = '#d7bde2'
COL_NOOP = '#95a5a6'

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


def draw_grid_bg(ax):
    for y in range(GRID_H):
        for x in range(GRID_W):
            ax.add_patch(Rectangle(
                (x, GRID_H - 1 - y), 1, 1,
                facecolor=COL_INACTIVE, edgecolor=COL_COMPUTE_BORDER,
                linewidth=0.7, zorder=1))
    ax.set_xlim(-0.15, GRID_W + 0.15)
    ax.set_ylim(-0.15, GRID_H + 0.15)
    ax.set_aspect('equal')
    ax.set_xticks(range(GRID_W))
    ax.set_yticks(range(GRID_H))
    # Label so (0,0) sits at top-left visually:
    #   axis y=3 (top)    -> logical y=0  -> phys 0
    #   axis y=0 (bottom) -> logical y=3  -> phys 4
    ax.set_yticklabels(
        [LOG_TO_PHYS_Y[GRID_H - 1 - i] for i in range(GRID_H)], fontsize=8)
    ax.set_xticklabels(range(GRID_W), fontsize=8)
    ax.tick_params(axis='both', colors='#666')
    ax.set_xlabel('core_x', fontsize=9)
    ax.set_ylabel('core_y (phys)', fontsize=9)
    for sp in ax.spines.values():
        sp.set_visible(False)


def paint(ax, gx, gy, color, border=COL_SHAPE_BORDER, lw=1.2):
    display_y = GRID_H - 1 - gy
    ax.add_patch(Rectangle(
        (gx, display_y), 1, 1,
        facecolor=color, edgecolor=border, linewidth=lw, zorder=3))


def draw_height_panel(ax):
    """Height-sharded: single sender (top_left), receivers = bbox rest."""
    for y in range(GRID_H):
        for x in range(GRID_W):
            paint(ax, x, y, COL_RECEIVER)
    paint(ax, 0, 0, COL_SINGLE_SENDER)


def draw_height_lshape_panel(ax):
    """Height-sharded L-shape variant: CORE_SET_14 = 5x2 top block +
    4x1 bottom row = 14 active cores.
      row 0: [S R R R R]   (5)   S = single sender
      row 1: [R R R R R]   (5)
      row 2: [R R R R .]   (4 active + 1 noop)
      row 3: (not part of bbox — all inactive)
    """
    # Top 3 rows: active area (bbox 5x3)
    for y in range(3):
        for x in range(GRID_W):
            paint(ax, x, y, COL_RECEIVER)
    # Row y=3 : not used at all — leave as inactive (drawn by grid_bg)
    # Noop cell at (4, 2)
    paint(ax, 4, 2, COL_NOOP)
    display_y = GRID_H - 1 - 2
    ax.text(4 + 0.5, display_y + 0.5, 'noop',
            ha='center', va='center', fontsize=8,
            color='white', fontweight='bold', zorder=4)
    # Sender at top-left
    paint(ax, 0, 0, COL_SINGLE_SENDER)


def draw_block_panel(ax):
    """Block-sharded 4x4: top row = W-sender, left col = A-sender,
    (0,0) = hot corner, rest = interior. Right col (x=4) unused."""
    W, H = 4, 4
    for dy in range(H):
        for dx in range(W):
            if dx == 0 and dy == 0:
                color = COL_HOT
            elif dy == 0:
                color = COL_W_SENDER
            elif dx == 0:
                color = COL_A_SENDER
            else:
                color = COL_INTERIOR
            paint(ax, dx, dy, color)


def draw_width_panel(ax):
    """Width-sharded (Matmul-like): weight is partitioned across cores
    (each core has a channel slice), activation is broadcast to all.
    So every core is a receiver of the broadcast + independent worker.
    Weight partition = local L1 read; no dominant sender in the same
    Height/Block sense. Depict entire bbox as broadcast destinations,
    with one broadcast-source marker (DRAM/producer) outside the grid."""
    for y in range(GRID_H):
        for x in range(GRID_W):
            paint(ax, x, y, COL_BROADCAST_DST)
    # Draw arrow from outside indicating activation broadcast to all cores
    ax.annotate('activation\nbroadcast\n(from DRAM)',
                xy=(2.5, GRID_H / 2), xycoords='data',
                xytext=(-1.7, GRID_H / 2), textcoords='data',
                fontsize=9, ha='center', va='center',
                color=COL_BROADCAST_SRC, fontweight='bold',
                arrowprops=dict(arrowstyle='-|>', color=COL_BROADCAST_SRC,
                                lw=2.0))


def main():
    fig = plt.figure(figsize=(19, 6.5))

    fig.text(0.5, 0.955,
             'Mcast structure comparison — Height vs Block vs Width sharding',
             ha='center', fontsize=15, fontweight='bold')
    fig.text(0.5, 0.915,
             'Same 5x4 compute grid; different sender/receiver roles per '
             'sharding scheme (source: sharded_program_factory.cpp)',
             ha='center', fontsize=10.5, color='#555', style='italic')

    panel_titles = [
        ('HEIGHT-sharded  (rectangular)',
         '1D weight mcast:  1 sender  ->  bbox rest\n'
         'no activation mcast (each core reads L1 shard)',
         '#3498db'),
        ('HEIGHT-sharded  (L-shape variant)',
         'noop cells inside bbox make the active\n'
         'shape non-rectangular  (e.g. CORE_SET_14)',
         '#3498db'),
        ('BLOCK-sharded',
         '2D mcast:  top row = weight sender,\n'
         'left col = activation sender.  (0,0) = hot corner',
         '#e67e22'),
        ('WIDTH-sharded  (Stage 4 Head only)',
         'Weight partitioned per core (K-dim slice);\n'
         'activation broadcast to all.  Separate factory.',
         '#c0392b'),
    ]

    draw_fns = [draw_height_panel, draw_height_lshape_panel,
                draw_block_panel, draw_width_panel]

    n_panels = 4
    panel_w = 0.20
    gap = 0.03
    total = n_panels * panel_w + (n_panels - 1) * gap
    left_margin = (1.0 - total) / 2

    for i, (fn, (title, sub, color)) in enumerate(zip(draw_fns, panel_titles)):
        left = left_margin + i * (panel_w + gap)
        ax = fig.add_axes([left, 0.22, panel_w, 0.55])
        draw_grid_bg(ax)
        fn(ax)

        fig.text(left + panel_w / 2, 0.855, title,
                 ha='center', fontsize=12, fontweight='bold',
                 color=color)
        fig.text(left + panel_w / 2, 0.805, sub,
                 ha='center', fontsize=9, color='#333')

    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor=COL_SINGLE_SENDER, edgecolor=COL_SHAPE_BORDER,
              label='Single mcast sender (Height)'),
        Patch(facecolor=COL_HOT, edgecolor=COL_SHAPE_BORDER,
              label='Hot corner (Block: W+A)'),
        Patch(facecolor=COL_W_SENDER, edgecolor=COL_SHAPE_BORDER,
              label='Weight sender row (Block)'),
        Patch(facecolor=COL_A_SENDER, edgecolor=COL_SHAPE_BORDER,
              label='Activation sender col (Block)'),
        Patch(facecolor=COL_INTERIOR, edgecolor=COL_SHAPE_BORDER,
              label='Receiver / interior'),
        Patch(facecolor=COL_BROADCAST_DST, edgecolor=COL_SHAPE_BORDER,
              label='Broadcast destination (Width)'),
        Patch(facecolor=COL_NOOP, edgecolor=COL_SHAPE_BORDER,
              label='noop cell (Height L-shape)'),
        Patch(facecolor=COL_INACTIVE, edgecolor=COL_COMPUTE_BORDER,
              label='Not used by op'),
    ]
    fig.legend(handles=legend_items, loc='lower center',
               bbox_to_anchor=(0.5, -0.02), ncol=8, fontsize=9,
               frameon=False)

    out = OUT_DIR / 'mcast_comparison_H_B_W.png'
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()

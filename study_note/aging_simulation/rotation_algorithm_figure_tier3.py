"""Paper-style figure showing Tier 3 aging-aware greedy algorithm.

Layout:
  TOP    : (a) current Vth  →  logic block (scoring formula + winner)  →  (c) after
  BOTTOM : 8 compact candidate thumbnails, each with worst-after score;
           winner wrapped in a green highlight box.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize

OUT_DIR = Path('C:/NPU_Challenge/study_note/aging_simulation/plots')

GRID_W = 5
GRID_H = 4
LOG_TO_PHYS_Y = {0: 0, 1: 1, 2: 2, 3: 4}
SHAPE_W = 4
SHAPE_H = 4

COL_INACTIVE = '#ecf0f1'
COL_COMPUTE_BORDER = '#7f8c8d'
COL_SHAPE_BORDER = '#d35400'
COL_HOT = '#c0392b'
COL_W_SENDER = '#e67e22'
COL_A_SENDER = '#f5b041'
COL_INTERIOR = '#f9e79f'
COL_WIN = '#27ae60'
COL_WIN_FILL = (0.15, 0.68, 0.38, 0.10)
COL_MAX_MARK = '#8e44ad'

REFLECTIONS = {
    'TL': {'hot': (0, 0),           'w_edge': 'top',    'a_edge': 'left'},
    'TR': {'hot': (SHAPE_W - 1, 0), 'w_edge': 'top',    'a_edge': 'right'},
    'BL': {'hot': (0, SHAPE_H - 1), 'w_edge': 'bottom', 'a_edge': 'left'},
    'BR': {'hot': (SHAPE_W - 1, SHAPE_H - 1),
                                    'w_edge': 'bottom', 'a_edge': 'right'},
}

DELTA_HOT = 0.007
DELTA_W_SENDER = 0.004
DELTA_A_SENDER = 0.003
DELTA_INTERIOR = 0.001

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


def _role(dx, dy, W, H, mode):
    info = REFLECTIONS[mode]
    hx, hy = info['hot']
    is_w_row = (info['w_edge'] == 'top' and dy == 0) or \
               (info['w_edge'] == 'bottom' and dy == H - 1)
    is_a_col = (info['a_edge'] == 'left' and dx == 0) or \
               (info['a_edge'] == 'right' and dx == W - 1)
    if (dx, dy) == (hx, hy):
        return 'hot'
    if is_w_row and is_a_col:
        return 'hot'
    if is_w_row:
        return 'w'
    if is_a_col:
        return 'a'
    return 'interior'


def role_color(dx, dy, W, H, mode):
    return {'hot': COL_HOT, 'w': COL_W_SENDER,
            'a': COL_A_SENDER, 'interior': COL_INTERIOR}[_role(dx, dy, W, H, mode)]


def role_delta(dx, dy, W, H, mode):
    return {'hot': DELTA_HOT, 'w': DELTA_W_SENDER,
            'a': DELTA_A_SENDER, 'interior': DELTA_INTERIOR}[_role(dx, dy, W, H, mode)]


def compute_delta_map(anchor, mode, W, H):
    d = np.zeros((GRID_H, GRID_W))
    for dy in range(H):
        for dx in range(W):
            gx = anchor[0] + dx
            gy = anchor[1] + dy
            d[gy, gx] = role_delta(dx, dy, W, H, mode)
    return d


def draw_vth_heatmap(ax, mat, vmin, vmax, title, annotate_max=True):
    cmap = plt.get_cmap('YlOrRd')
    norm = Normalize(vmin=vmin, vmax=vmax)
    im = ax.imshow(mat, cmap=cmap, norm=norm, aspect='equal')
    for (i, j), v in np.ndenumerate(mat):
        text_color = 'white' if v > 0.55 * vmax else '#111'
        ax.text(j, i, f'{v*1000:.1f}', ha='center', va='center',
                fontsize=9.5, color=text_color, fontweight='medium')
    if annotate_max:
        i, j = np.unravel_index(np.argmax(mat), mat.shape)
        ax.add_patch(Rectangle(
            (j - 0.5, i - 0.5), 1, 1,
            edgecolor=COL_MAX_MARK, linewidth=2.5, facecolor='none',
            zorder=5))
    ax.set_xticks(range(GRID_W))
    ax.set_yticks(range(GRID_H))
    ax.set_yticklabels([LOG_TO_PHYS_Y[y] for y in range(GRID_H)])
    ax.set_xlabel('core_x', fontsize=9)
    ax.set_ylabel('core_y (phys)', fontsize=9)
    ax.set_title(title, fontsize=10.5, pad=6)


def draw_candidate_thumb(ax, anchor, mode):
    for y in range(GRID_H):
        for x in range(GRID_W):
            ax.add_patch(Rectangle(
                (x, GRID_H - 1 - y), 1, 1,
                facecolor=COL_INACTIVE, edgecolor=COL_COMPUTE_BORDER,
                linewidth=0.6, zorder=1))
    for dy in range(SHAPE_H):
        for dx in range(SHAPE_W):
            gx = anchor[0] + dx
            gy = anchor[1] + dy
            color = role_color(dx, dy, SHAPE_W, SHAPE_H, mode)
            display_y = GRID_H - 1 - gy
            ax.add_patch(Rectangle(
                (gx, display_y), 1, 1,
                facecolor=color, edgecolor=COL_SHAPE_BORDER,
                linewidth=0.9, zorder=3))
    ax.set_xlim(-0.15, GRID_W + 0.15)
    ax.set_ylim(-0.15, GRID_H + 0.15)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def main():
    # Synthetic mid-schedule state; constructed so winner is clear.
    current = np.array([
        [0.018, 0.024, 0.020, 0.017, 0.014],
        [0.024, 0.020, 0.015, 0.013, 0.011],
        [0.020, 0.015, 0.013, 0.011, 0.009],
        [0.025, 0.013, 0.011, 0.009, 0.007],
    ])
    current_worst = float(current.max())

    anchors = [(0, 0), (1, 0)]
    modes = ['TL', 'TR', 'BL', 'BR']
    candidates = []
    for anchor in anchors:
        for mode in modes:
            d = compute_delta_map(anchor, mode, SHAPE_W, SHAPE_H)
            after = current + d
            candidates.append({
                'anchor': anchor, 'mode': mode,
                'delta': d, 'after': after, 'worst': float(after.max()),
            })
    winner = min(candidates, key=lambda c: c['worst'])
    sorted_cands = sorted(candidates, key=lambda c: c['worst'])

    print('=== Candidate ranking ===')
    print(f'Current worst: {current_worst * 1000:.3f} mV')
    for c in sorted_cands:
        mark = '  <-- WINNER' if c is winner else ''
        print(f'  ({c["anchor"][0]},{c["anchor"][1]}) + {c["mode"]}: '
              f'worst = {c["worst"] * 1000:.3f} mV{mark}')

    # ================================================
    fig = plt.figure(figsize=(17.5, 10))

    fig.text(0.5, 0.975,
             'Tier 3 rotation: aging-aware greedy (min-worst config selection)',
             ha='center', fontsize=15, fontweight='bold')
    fig.text(0.5, 0.947,
             'For each op: enumerate all valid configs, predict per-core Vth '
             'after placement, pick argmin(worst-core).',
             ha='center', fontsize=10.5, color='#555', style='italic')

    vmax = max(current.max(), max(c['worst'] for c in candidates))
    vmin = 0

    # ================= TOP ROW =================
    # (a) Current
    ax_a = fig.add_axes([0.035, 0.55, 0.19, 0.36])
    draw_vth_heatmap(
        ax_a, current, vmin, vmax,
        f'(a) Current per-core Vth (mV)\nworst = {current_worst * 1000:.2f} mV',
        annotate_max=True)

    # Middle band: arrow flow + winner summary only
    mid = 0.505
    arrow_y = 0.735

    # Arrows spanning most of the middle width
    fig.text(0.255, arrow_y, r'$\longrightarrow$', ha='center', va='center',
             fontsize=40, color='#3498db', fontweight='bold')
    fig.text(0.755, arrow_y, r'$\longrightarrow$', ha='center', va='center',
             fontsize=40, color=COL_WIN, fontweight='bold')

    # Center label between the two arrows
    fig.text(mid, arrow_y + 0.055,
             'argmin worst-after Vth', ha='center', va='center',
             fontsize=14, fontweight='bold', color='#2c3e50')
    fig.text(mid, arrow_y + 0.005,
             f'Winner: ({winner["anchor"][0]},{winner["anchor"][1]}) + {winner["mode"]}',
             ha='center', va='center', fontsize=14,
             color=COL_WIN, fontweight='bold')
    fig.text(mid, arrow_y - 0.045,
             f'worst-after = {winner["worst"] * 1000:.2f} mV   '
             f'(Δ = {(winner["worst"] - current_worst) * 1000:+.2f} mV vs current)',
             ha='center', va='center', fontsize=11,
             color=COL_WIN, fontweight='medium')
    fig.text(mid, arrow_y - 0.090,
             'hot corner lands on lowest-Vth region',
             ha='center', va='center', fontsize=10,
             style='italic', color='#666')

    # (c) After
    ax_c = fig.add_axes([0.785, 0.55, 0.19, 0.36])
    draw_vth_heatmap(
        ax_c, winner['after'], vmin, vmax,
        f'(c) After chosen config (mV)\n'
        f'new worst = {winner["worst"] * 1000:.2f} mV',
        annotate_max=True)

    # ================= BOTTOM: 8 candidates =================
    fig.text(0.5, 0.485,
             '(b) 8 candidates and their scores  —  green box = argmin',
             ha='center', fontsize=13, fontweight='bold')

    panel_w = 0.19
    panel_h = 0.14
    gap_x = 0.025
    n_cols = 4
    total_w = n_cols * panel_w + (n_cols - 1) * gap_x
    left_margin = (1.0 - total_w) / 2
    # Row 0: top, panel bottom = 0.28
    # Row 1: bot, panel bottom = 0.055
    row_bottoms = [0.28, 0.055]

    for i, cand in enumerate(candidates):
        row = i // 4
        col = i % 4
        left = left_margin + col * (panel_w + gap_x)
        bottom = row_bottoms[row]

        ax = fig.add_axes([left, bottom, panel_w, panel_h])
        draw_candidate_thumb(ax, cand['anchor'], cand['mode'])

        is_winner = cand is winner
        tc = COL_WIN if is_winner else '#333'
        tw = 'bold' if is_winner else 'medium'

        prefix = 'CHOSEN >>  ' if is_winner else ''
        ax.set_title(
            f'{prefix}({cand["anchor"][0]},{cand["anchor"][1]}) + {cand["mode"]}',
            fontsize=10.5, color=tc, fontweight=tw, pad=3)

        center_x = left + panel_w / 2
        fig.text(center_x, bottom - 0.014,
                 f'worst-after = {cand["worst"] * 1000:.2f} mV',
                 ha='center', va='top', fontsize=10,
                 color=tc, fontweight=tw)

        if is_winner:
            pad = 0.010
            box_left = left - pad
            box_top = bottom + panel_h + 0.032
            box_bottom = bottom - 0.040
            box_w = panel_w + 2 * pad
            box_h = box_top - box_bottom
            fig.add_artist(Rectangle(
                (box_left, box_bottom), box_w, box_h,
                edgecolor=COL_WIN, linewidth=2.4,
                facecolor=COL_WIN_FILL, zorder=0))

    # ================= Legend =================
    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor=COL_HOT, edgecolor=COL_SHAPE_BORDER,
              label='Hot corner (W+A sender)'),
        Patch(facecolor=COL_W_SENDER, edgecolor=COL_SHAPE_BORDER,
              label='Weight sender row'),
        Patch(facecolor=COL_A_SENDER, edgecolor=COL_SHAPE_BORDER,
              label='Activation sender col'),
        Patch(facecolor=COL_INTERIOR, edgecolor=COL_SHAPE_BORDER,
              label='Interior'),
        Patch(facecolor=COL_INACTIVE, edgecolor=COL_COMPUTE_BORDER,
              label='Not used by this op'),
        Patch(facecolor='none', edgecolor=COL_MAX_MARK, linewidth=2,
              label='Worst-core cell'),
    ]
    fig.legend(handles=legend_items, loc='lower center',
               bbox_to_anchor=(0.5, -0.02), ncol=6, fontsize=9.5,
               frameon=False)

    out = OUT_DIR / 'algorithm_tier3_illustration.png'
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f'\nWrote {out}')


if __name__ == '__main__':
    main()

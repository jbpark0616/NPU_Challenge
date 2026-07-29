"""Rotation tier summary table — worst-core & spread reduction per tier
(matches the clean style of mcast_comparison_table.py).
"""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT_DIR = Path('C:/NPU_Challenge/study_note/aging_simulation/plots')

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 11,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

COL_HEADER_BG = '#2c3e50'
COL_HEADER_TEXT = 'white'
COL_ROWLABEL_BG = '#f4f6f7'
COL_CELL_BG = 'white'
COL_BORDER = '#d0d3d4'
COL_BEST_BG = '#eafaf1'
COL_DELTA = '#27ae60'
COL_TEXT = '#333'

# Numbers extracted from rotation_tier3_per_core.csv
# NIU = total_noc  (injected + consumed bytes on the NoC)
# SRAM = consumed_bytes  (L1 write side, the aging-driving component)
TIER_ROWS = [
    ('Baseline',  '14.67 MB',                 '13.08×',                '12.65 MB',                '74.14×'),
    ('Tier 1',    '12.74 MB   (−13%)',        '10.32×   (−21%)',       '10.14 MB   (−20%)',       '59.43×   (−20%)'),
    ('Tier 2',    '9.55 MB   (−35%)',         '5.61×   (−57%)',        '6.93 MB   (−45%)',        '41.01×   (−45%)'),
    ('Tier 3',    '6.76 MB   (−54%)',         '5.25×   (−60%)',        '4.82 MB   (−62%)',        '28.05×   (−62%)'),
]

HEADER = ('', 'NIU  worst-core', 'NIU  spread', 'SRAM  worst-core', 'SRAM  spread')

COL_WIDTHS = [0.13, 0.20, 0.20, 0.20, 0.20]


def main():
    fig = plt.figure(figsize=(15, 4.4))

    fig.text(0.5, 0.945,
             'Rotation tier summary — worst-core & spread reduction vs baseline',
             ha='center', fontsize=15, fontweight='bold')
    fig.text(0.5, 0.895,
             'Source: rotation_tier{1,2,3}_per_core.csv  '
             '(NIU = total NoC bytes,  SRAM = L1 consumed bytes)',
             ha='center', fontsize=10, color='#555', style='italic')

    n_rows = 1 + len(TIER_ROWS)          # header + tiers
    total_w = sum(COL_WIDTHS)
    left_start = (1.0 - total_w) / 2

    row_h = 0.14
    top_y = 0.72

    rows = [HEADER] + TIER_ROWS
    n_data = len(TIER_ROWS)

    for r, row in enumerate(rows):
        bottom = top_y - r * row_h
        x = left_start
        is_best = (r == n_data)          # last data row (Tier 3)

        for c, cell in enumerate(row):
            cw = COL_WIDTHS[c]

            if r == 0:
                bg = COL_HEADER_BG
                text_color = COL_HEADER_TEXT
                fw = 'bold'
                fs = 11.5
            else:
                if c == 0:
                    bg = COL_ROWLABEL_BG if not is_best else COL_BEST_BG
                    text_color = '#2c3e50'
                    fw = 'bold'
                else:
                    bg = COL_CELL_BG if not is_best else COL_BEST_BG
                    text_color = COL_TEXT
                    fw = 'bold' if is_best else 'normal'
                fs = 11

            fig.add_artist(Rectangle(
                (x, bottom), cw, row_h,
                facecolor=bg, edgecolor=COL_BORDER,
                linewidth=0.8, zorder=1))

            # For data cells with "(−XX%)" delta, colorize just the delta text.
            if r > 0 and c > 0 and '(' in cell:
                main_part, delta_part = cell.split('(', 1)
                delta_part = '(' + delta_part
                fig.text(x + cw / 2 - 0.028, bottom + row_h / 2,
                         main_part.strip(),
                         ha='right', va='center', fontsize=fs,
                         color=text_color, fontweight=fw, zorder=2)
                fig.text(x + cw / 2 + 0.006, bottom + row_h / 2,
                         delta_part,
                         ha='left', va='center', fontsize=fs - 0.5,
                         color=COL_DELTA, fontweight='bold', zorder=2)
            else:
                fig.text(x + cw / 2, bottom + row_h / 2, cell,
                         ha='center', va='center', fontsize=fs,
                         color=text_color, fontweight=fw, zorder=2)
            x += cw

    out = OUT_DIR / 'rotation_tier_summary_table.png'
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()

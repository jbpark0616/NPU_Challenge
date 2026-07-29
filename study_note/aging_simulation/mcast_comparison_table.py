"""Structural comparison table (PPT-ready standalone slide).

Rows: mcast property axes.  Columns: Height / Block / Width.
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
COL_YES = '#27ae60'
COL_NO = '#c0392b'

TABLE_ROWS = [
    ('',                 'Height',                      'Block',                       'Width'),
    ('Mcast pattern',    '1D weight (single sender)',   '2D (weight + activation)',    'broadcast only'),
    ('Sender location',  'top-left core',               'top row + left col',          '(no fan-out sender)'),
    ('Rotation freedom', 'anchor shift + L-shape',      'anchor shift + 4-reflection', 'N/A'),
    ('In rotation scope?', 'YES',                       'YES',                         'NO'),
]

COL_WIDTHS = [0.20, 0.22, 0.24, 0.20]


def main():
    fig = plt.figure(figsize=(14, 4.8))

    fig.text(0.5, 0.94,
             'Sharding scheme comparison',
             ha='center', fontsize=15, fontweight='bold')

    n_rows = len(TABLE_ROWS)
    total_w = sum(COL_WIDTHS)
    left_start = (1.0 - total_w) / 2

    row_h = 0.11
    top_y = 0.76

    for r, row in enumerate(TABLE_ROWS):
        bottom = top_y - r * row_h
        x = left_start
        for c, cell in enumerate(row):
            cw = COL_WIDTHS[c]

            if r == 0:
                bg = COL_HEADER_BG
                text_color = COL_HEADER_TEXT
                fw = 'bold'
                fs = 12
            else:
                if c == 0:
                    bg = COL_ROWLABEL_BG
                    text_color = '#2c3e50'
                    fw = 'bold'
                else:
                    bg = COL_CELL_BG
                    if r == n_rows - 1:
                        text_color = COL_YES if cell == 'YES' else COL_NO
                        fw = 'bold'
                    else:
                        text_color = '#333'
                        fw = 'normal'
                fs = 11

            fig.add_artist(Rectangle(
                (x, bottom), cw, row_h,
                facecolor=bg, edgecolor=COL_BORDER,
                linewidth=0.8, zorder=1))
            fig.text(x + cw / 2, bottom + row_h / 2, cell,
                     ha='center', va='center', fontsize=fs,
                     color=text_color, fontweight=fw, zorder=2)
            x += cw

    out = OUT_DIR / 'mcast_comparison_table.png'
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()

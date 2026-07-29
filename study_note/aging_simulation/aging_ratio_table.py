"""Worst/best aging ratio table per axis (matches the mcast/tier table style).

Shows for each axis (NIU / SRAM / Compute):
  - raw activity ratio (worst/best of the measured metric)
  - NBTI aging ratio     = raw^(1/6)   (Chen NBTI: u^(1/6) dependence)
  - HCI  aging ratio     = raw         (Chen HCI:  u^1  dependence)
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
COL_HCI_HIGHLIGHT = '#fef5f0'   # very light warning tone for HCI col
COL_HCI_TEXT = '#c0392b'
COL_TEXT = '#333'

# Numbers computed from per_core.csv (Tracy) + per_core_compute.csv (device log)
DATA_ROWS = [
    ('NIU',     '13.08×', '1.54×',  '13.08×'),
    ('SRAM',    '74.14×', '2.05×',  '74.14×'),
    ('Compute', '30.35×', '1.77×',  '30.35×'),
]

HEADER = ('Axis',
          'Raw activity\n(worst / best)',
          'NBTI aging ratio\n= raw^(1/6)',
          'HCI aging ratio\n= raw')

COL_WIDTHS = [0.14, 0.22, 0.22, 0.22]


def main():
    fig = plt.figure(figsize=(13, 4.5))

    fig.text(0.5, 0.94,
             'Worst / best aging ratio per axis  (baseline, ResNet50 batch=1)',
             ha='center', fontsize=15, fontweight='bold')
    fig.text(0.5, 0.885,
             'NBTI compresses activity imbalance by 6th-root;  '
             'HCI carries it linearly → HCI reveals the true aging skew.',
             ha='center', fontsize=10, color='#555', style='italic')

    total_w = sum(COL_WIDTHS)
    left_start = (1.0 - total_w) / 2

    row_h = 0.135
    top_y = 0.68

    rows = [HEADER] + DATA_ROWS
    for r, row in enumerate(rows):
        bottom = top_y - r * row_h
        x = left_start
        for c, cell in enumerate(row):
            cw = COL_WIDTHS[c]

            if r == 0:
                bg = COL_HEADER_BG
                text_color = COL_HEADER_TEXT
                fw = 'bold'
                fs = 11
            else:
                if c == 0:
                    bg = COL_ROWLABEL_BG
                    text_color = '#2c3e50'
                    fw = 'bold'
                elif c == 3:                       # HCI column → highlight
                    bg = COL_HCI_HIGHLIGHT
                    text_color = COL_HCI_TEXT
                    fw = 'bold'
                else:
                    bg = COL_CELL_BG
                    text_color = COL_TEXT
                    fw = 'normal'
                fs = 12

            fig.add_artist(Rectangle(
                (x, bottom), cw, row_h,
                facecolor=bg, edgecolor=COL_BORDER,
                linewidth=0.8, zorder=1))
            fig.text(x + cw / 2, bottom + row_h / 2, cell,
                     ha='center', va='center', fontsize=fs,
                     color=text_color, fontweight=fw, zorder=2)
            x += cw

    fig.text(0.5, 0.05,
             'Read: SRAM worst-core wears 74× faster than SRAM best-core '
             '(HCI-dominant regime).  System lifetime = worst-core lifetime.',
             ha='center', fontsize=10.5, color='#2c3e50', style='italic')

    out = OUT_DIR / 'aging_ratio_table.png'
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()

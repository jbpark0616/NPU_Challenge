"""Blackhole NoC router figure — paper style (KAIST-thesis alike).
Left: tile-level NoC0/NoC1 mini view (Tensix core body omitted).
Right: zoomed NoC0 router — VC buffers + Switch + ★ BRCST_EXCLUDE control register.
Style: sharp black rectangles, thin borders, dashed callout, minimal color.
"""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from lxml import etree

# ---------- palette (paper style, monochrome + 2 traffic colors + 1 accent) ----------
BLACK  = RGBColor(0x00, 0x00, 0x00)
GRAY   = RGBColor(0x80, 0x80, 0x80)
LGRAY  = RGBColor(0xCC, 0xCC, 0xCC)
WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
RED    = RGBColor(0xD0, 0x21, 0x2C)   # NoC0
GREEN  = RGBColor(0x1E, 0x8A, 0x2E)   # NoC1
LBLUE  = RGBColor(0xEA, 0xF1, 0xFB)   # faint tile background
VC_BG  = RGBColor(0xE8, 0xF0, 0xFB)   # VC bank background
ACCENT = RGBColor(0xC0, 0x39, 0x2B)   # BRCST_EXCLUDE
ACC_BG = RGBColor(0xFD, 0xEA, 0xE7)

# ---------- slide ----------
prs = Presentation()
prs.slide_width  = Inches(13.333)
prs.slide_height = Inches(7.5)
slide = prs.slides.add_slide(prs.slide_layouts[6])

# ---------- primitives ----------
def rect(x, y, w, h, *, fill=WHITE, stroke=BLACK, sw=1.0, rounded=False, dashed=False, adj=0.05):
    kind = MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE
    s = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid(); s.fill.fore_color.rgb = fill
    if stroke is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = stroke
        s.line.width = Pt(sw)
        if dashed:
            ln = s.line._get_or_add_ln()
            prst = ln.find(qn('a:prstDash'))
            if prst is None:
                prst = etree.SubElement(ln, qn('a:prstDash'))
            prst.set('val', 'dash')
    if rounded:
        try:
            s.adjustments[0] = adj
        except Exception:
            pass
    s.shadow.inherit = False
    return s

def _run(r, *, size, bold, color, name='Calibri'):
    r.font.name = name; r.font.size = Pt(size)
    r.font.bold = bold; r.font.color.rgb = color

def label(shape, text, *, size=11, bold=True, color=BLACK,
          align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, pad=0.04):
    tf = shape.text_frame
    tf.margin_left = Inches(pad); tf.margin_right = Inches(pad)
    tf.margin_top = Inches(pad*0.4); tf.margin_bottom = Inches(pad*0.4)
    tf.word_wrap = True; tf.vertical_anchor = anchor
    p = tf.paragraphs[0]; p.alignment = align
    p.text = text if text else ' '
    _run(p.runs[0], size=size, bold=bold, color=color)
    return tf

def mlabel(shape, lines, *, sizes, bolds=None, colors=None,
           align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, pad=0.04):
    n = len(lines)
    bolds  = bolds  or [True]*n
    colors = colors or [BLACK]*n
    tf = shape.text_frame
    tf.margin_left = Inches(pad); tf.margin_right = Inches(pad)
    tf.margin_top = Inches(pad*0.4); tf.margin_bottom = Inches(pad*0.4)
    tf.word_wrap = True; tf.vertical_anchor = anchor
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.text = ln if ln else ' '
        _run(p.runs[0], size=sizes[i], bold=bolds[i], color=colors[i])

def arrow(x1, y1, x2, y2, *, color=BLACK, width=1.0, dashed=False, head='triangle'):
    c = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1),
        Inches(x2 - x1), Inches(y2 - y1))
    c.line.color.rgb = color
    c.line.width = Pt(width)
    ln = c.line._get_or_add_ln()
    tail = etree.SubElement(ln, qn('a:tailEnd'))
    tail.set('type', head); tail.set('w', 'med'); tail.set('len', 'med')
    if dashed:
        prst = ln.find(qn('a:prstDash'))
        if prst is None:
            prst = etree.SubElement(ln, qn('a:prstDash'))
        prst.set('val', 'dash')
    return c

def plain_line(x1, y1, x2, y2, *, color=GRAY, width=0.75, dashed=True):
    c = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1),
        Inches(x2 - x1), Inches(y2 - y1))
    c.line.color.rgb = color
    c.line.width = Pt(width)
    if dashed:
        ln = c.line._get_or_add_ln()
        prst = ln.find(qn('a:prstDash'))
        if prst is None:
            prst = etree.SubElement(ln, qn('a:prstDash'))
        prst.set('val', 'dash')
    return c


# ==========================================================
# LEGEND (top)
# ==========================================================
arrow(0.55, 0.32, 0.95, 0.32, color=RED, width=2.5)
lg = rect(1.0, 0.20, 1.7, 0.28, stroke=None); label(lg, ': NoC 0 traffic', size=10, bold=False, align=PP_ALIGN.LEFT, pad=0.02)
arrow(2.75, 0.32, 3.15, 0.32, color=GREEN, width=2.5)
lg = rect(3.2, 0.20, 1.7, 0.28, stroke=None); label(lg, ': NoC 1 traffic', size=10, bold=False, align=PP_ALIGN.LEFT, pad=0.02)
plain_line(4.9, 0.32, 5.35, 0.32, color=GRAY, width=1.0, dashed=True)
lg = rect(5.4, 0.20, 2.0, 0.28, stroke=None); label(lg, ': zoom callout', size=10, bold=False, align=PP_ALIGN.LEFT, pad=0.02)


# ==========================================================
# LEFT — tile-level NoC0 / NoC1 mini view
# ==========================================================
TX, TY, TW, TH = 0.7, 1.3, 3.8, 5.4
tile = rect(TX, TY, TW, TH, fill=LBLUE, stroke=BLACK, sw=1.1)

# 'Tensix Core (body omitted)' faint interior
inner = rect(TX + 0.35, TY + 0.35, TW - 0.7, TH - 2.15,
             fill=WHITE, stroke=LGRAY, sw=0.6, dashed=True)
mlabel(inner,
       ['Tensix Core', '', '(compute · L1 SRAM', 'body omitted)'],
       sizes=[13, 6, 10, 10],
       bolds=[True, False, False, False],
       colors=[GRAY, GRAY, GRAY, GRAY])

# NoC1 & NoC0 boxes near bottom
NB_Y  = TY + TH - 1.5
NB_H  = 0.7
NB_W  = 1.15
nb1_x = TX + 0.45
nb0_x = TX + TW - 0.45 - NB_W
nb1 = rect(nb1_x, NB_Y, NB_W, NB_H, stroke=BLACK, sw=1.2)
label(nb1, 'NoC 1', size=14, bold=True)
nb0 = rect(nb0_x, NB_Y, NB_W, NB_H, stroke=BLACK, sw=1.2)
label(nb0, 'NoC 0', size=14, bold=True)

# 'Tensix Core' bottom label
core_lbl = rect(TX, TY + TH + 0.02, TW, 0.32, stroke=None)
label(core_lbl, 'Tensix Core', size=13, bold=True, align=PP_ALIGN.LEFT, pad=0.1)

# --- directional arrows out of tile ---
# NoC 1 (green): W  (left)  and  N  (up)  — bidirectional pair
# NoC 0 (red)  : E  (right) and  S  (down) — bidirectional pair

# NoC1 : Left side (W)  — one out, one in
arrow(nb1_x, NB_Y + NB_H*0.35, TX - 0.25, NB_Y + NB_H*0.35, color=GREEN, width=2.25)
arrow(TX - 0.25, NB_Y + NB_H*0.70, nb1_x, NB_Y + NB_H*0.70, color=GREEN, width=2.25)
# NoC1 : Top side (N) from left half column
mid_n = nb1_x + NB_W*0.5
arrow(mid_n - 0.10, NB_Y, mid_n - 0.10, TY - 0.25, color=GREEN, width=2.25)
arrow(mid_n + 0.10, TY - 0.25, mid_n + 0.10, NB_Y, color=GREEN, width=2.25)

# NoC0 : Right side (E)
arrow(nb0_x + NB_W, NB_Y + NB_H*0.35, TX + TW + 0.25, NB_Y + NB_H*0.35, color=RED, width=2.25)
arrow(TX + TW + 0.25, NB_Y + NB_H*0.70, nb0_x + NB_W, NB_Y + NB_H*0.70, color=RED, width=2.25)
# NoC0 : Bottom side (S)
mid_s = nb0_x + NB_W*0.5
arrow(mid_s - 0.10, NB_Y + NB_H, mid_s - 0.10, TY + TH + 0.55, color=RED, width=2.25)
arrow(mid_s + 0.10, TY + TH + 0.55, mid_s + 0.10, NB_Y + NB_H, color=RED, width=2.25)

# small compass labels around tile
def txt_at(x, y, w, h, s, sz=9, b=True, c=BLACK):
    r = rect(x, y, w, h, stroke=None); label(r, s, size=sz, bold=b, color=c)

txt_at(TX - 0.45, NB_Y - 0.18, 0.4, 0.2, 'W', c=GREEN)
txt_at(mid_n - 0.3, TY - 0.4, 0.6, 0.2, 'N', c=GREEN)
txt_at(TX + TW + 0.05, NB_Y - 0.18, 0.4, 0.2, 'E', c=RED)
txt_at(mid_s - 0.2, TY + TH + 0.55, 0.4, 0.2, 'S', c=RED)


# ==========================================================
# CALLOUT — dashed lines from NoC0 to the router zoom
# ==========================================================
zoom_x = 5.7
plain_line(nb0_x + NB_W, NB_Y + 0.05,  zoom_x, 1.55, color=GRAY, width=0.9, dashed=True)
plain_line(nb0_x + NB_W, NB_Y + NB_H,  zoom_x, 6.5,  color=GRAY, width=0.9, dashed=True)


# ==========================================================
# RIGHT — zoomed NoC 0 router
# ==========================================================
ZX, ZY, ZW, ZH = 5.7, 1.55, 7.35, 4.95
zoom = rect(ZX, ZY, ZW, ZH, fill=WHITE, stroke=BLACK, sw=1.1)

# Zoom title (small tag top-left inside)
tag = rect(ZX + 0.08, ZY + 0.08, 1.9, 0.28, stroke=None)
label(tag, 'NoC 0 Router (zoom)', size=11, bold=True, color=BLACK, align=PP_ALIGN.LEFT, pad=0.02)

# --- input port labels (left of zoom) ---
in_labels = [('PE',    ZY + 1.15),
             ('North', ZY + 2.15),
             ('West',  ZY + 3.15)]
for name, py in in_labels:
    r = rect(ZX - 0.75, py - 0.13, 0.7, 0.26, stroke=None)
    label(r, name, size=11, bold=True, align=PP_ALIGN.RIGHT, pad=0.02)
    arrow(ZX - 0.05, py, ZX + 0.85, py, color=BLACK, width=1.5)

# 32B tag near West input
tag32 = rect(ZX - 0.15, ZY + 3.3, 0.7, 0.22, stroke=None)
label(tag32, '32 B', size=10, bold=True, color=BLACK, align=PP_ALIGN.LEFT, pad=0.02)

# --- VC bank (stacked banks, one per input port) ---
VB_X, VB_Y = ZX + 0.85, ZY + 0.55
VB_W, VB_H = 2.4, 3.35

# 3 stacked bank boxes with slight offset (isometric feel like the reference)
def vc_bank(x, y):
    outer = rect(x, y, VB_W, VB_H*0.85, fill=VC_BG, stroke=BLACK, sw=0.9)
    # VC list inside
    inner_x = x + 0.15; inner_y = y + 0.15
    row_h = 0.35
    labels = [('VC 0',  inner_y + 0*row_h),
              ('VC 1',  inner_y + 1*row_h),
              ('...',   inner_y + 2*row_h),
              ('VC 15', inner_y + 3*row_h)]
    for name, ry in labels:
        cell = rect(inner_x, ry, VB_W - 0.30, 0.28, fill=WHITE, stroke=BLACK, sw=0.5)
        label(cell, name, size=9, bold=False, align=PP_ALIGN.LEFT, pad=0.10)
        # little FIFO tick marks on the right
        for k in range(4):
            rect(inner_x + VB_W - 0.55 + k*0.09, ry + 0.06, 0.06, 0.16,
                 fill=WHITE, stroke=BLACK, sw=0.4)
    return outer

# stacked (top slight offset)
vc_bank(VB_X + 0.20, VB_Y + 0.00)
vc_bank(VB_X + 0.10, VB_Y + 0.10)
front = vc_bank(VB_X + 0.00, VB_Y + 0.20)

# --- Switch (big X crossbar box) ---
SW_X, SW_Y, SW_W, SW_H = ZX + 4.05, ZY + 0.95, 1.8, 2.65
sw = rect(SW_X, SW_Y, SW_W, SW_H, fill=WHITE, stroke=BLACK, sw=1.1)
# draw the X inside
plain_line(SW_X + 0.15, SW_Y + 0.15, SW_X + SW_W - 0.15, SW_Y + SW_H - 0.15,
           color=BLACK, width=1.4, dashed=False)
plain_line(SW_X + SW_W - 0.15, SW_Y + 0.15, SW_X + 0.15, SW_Y + SW_H - 0.15,
           color=BLACK, width=1.4, dashed=False)
sw_lbl = rect(SW_X, SW_Y + SW_H + 0.05, SW_W, 0.28, stroke=None)
label(sw_lbl, 'Switch', size=11, bold=True)

# VC banks -> Switch : 3 arrows (one per bank)
arrow(VB_X + VB_W + 0.02, ZY + 1.15, SW_X, SW_Y + 0.35, color=BLACK, width=1.2)
arrow(VB_X + VB_W + 0.02, ZY + 2.15, SW_X, SW_Y + SW_H*0.5, color=BLACK, width=1.2)
arrow(VB_X + VB_W + 0.02, ZY + 3.15, SW_X, SW_Y + SW_H - 0.35, color=BLACK, width=1.2)

# --- output port arrows (right of switch) ---
out_labels = [('PE',    SW_Y + 0.45),
              ('South', SW_Y + SW_H*0.5),
              ('East',  SW_Y + SW_H - 0.45)]
for name, py in out_labels:
    arrow(SW_X + SW_W, py, SW_X + SW_W + 0.85, py, color=BLACK, width=1.5)
    r = rect(SW_X + SW_W + 0.90, py - 0.14, 0.6, 0.28, stroke=None)
    label(r, name, size=11, bold=True, align=PP_ALIGN.LEFT, pad=0.02)

# ==========================================================
# ★ BRCST_EXCLUDE control register — new-in-Blackhole highlight
# ==========================================================
BE_X, BE_Y, BE_W, BE_H = SW_X - 0.35, ZY + ZH - 1.15, SW_W + 0.7, 0.85
be_wrap = rect(BE_X, BE_Y, BE_W, BE_H, fill=ACC_BG, stroke=ACCENT, sw=1.5)
be_hdr = rect(BE_X, BE_Y, BE_W, 0.24, fill=ACCENT, stroke=ACCENT)
label(be_hdr, '★  New in Blackhole', size=9, bold=True, color=WHITE, align=PP_ALIGN.LEFT, pad=0.08)
be_body = rect(BE_X + 0.08, BE_Y + 0.28, BE_W - 0.16, BE_H - 0.34,
               fill=WHITE, stroke=ACCENT, sw=1.0)
mlabel(be_body,
       ['NOC_BRCST_EXCLUDE',
        '(offset 0x2C, 24-bit mask)'],
       sizes=[12, 9],
       bolds=[True, False],
       colors=[ACCENT, ACCENT])

# BRCST_EXCLUDE -> Switch (control input from below)
arrow(BE_X + BE_W/2, BE_Y, BE_X + BE_W/2, SW_Y + SW_H,
      color=ACCENT, width=1.4, dashed=True)

# small annotation next to the control arrow
ann = rect(BE_X + BE_W/2 + 0.05, (BE_Y + SW_Y + SW_H)/2 - 0.15, 1.8, 0.28, stroke=None)
label(ann, 'per-mcast destination mask',
      size=9, bold=False, color=ACCENT, align=PP_ALIGN.LEFT, pad=0.02)


# ==========================================================
# CAPTION (below zoom)
# ==========================================================
cap = rect(0.7, 6.85, 12.6, 0.55, stroke=None)
mlabel(cap,
       ['Fig. — NoC subsystem of a Blackhole Tensix tile.  Left: two independent NoCs (NoC 0 uses X-Y '
        'routing, NoC 1 uses Y-X, disjoint physical channels).  Right: NoC 0 router internals — per-input VC '
        'buffers feed a 5-port switch.  ★ BRCST_EXCLUDE (Blackhole-new, absent in Wormhole) supplies a '
        '24-bit destination-drop mask to the switch\u2019s broadcast logic.'],
       sizes=[9], bolds=[False], colors=[BLACK],
       align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, pad=0.05)


# ==========================================================
# SAVE
# ==========================================================
out = r'C:\NPU_Challenge\study_note\figures\noc_router_figure.pptx'
prs.save(out)
print('OK ->', out)

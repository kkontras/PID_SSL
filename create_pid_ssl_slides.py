"""
create_pid_ssl_slides.py
========================
Creates a two-slide PowerPoint deck explaining the PID-SAR-3 data generation process.

Slide 1 — Schematic of the data generation pipeline
Slide 2 — Full mathematical description of each PID atom type

Run from the project root:
    python create_pid_ssl_slides.py
Output: pid_ssl_data_generation.pptx
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt
import pptx.util as util

# ── colour palette ────────────────────────────────────────────────────────────
C_BG        = RGBColor(0x1A, 0x1A, 0x2E)   # deep navy background
C_TITLE     = RGBColor(0xE0, 0xE0, 0xFF)   # near-white title
C_SUBTITLE  = RGBColor(0xA0, 0xC4, 0xFF)   # light blue subtitle
C_UNIQUE    = RGBColor(0x54, 0xA2, 0x4B)   # green  — Unique
C_REDUND    = RGBColor(0x4C, 0x78, 0xA8)   # blue   — Redundancy
C_SYNERGY   = RGBColor(0xE4, 0x57, 0x56)   # red    — Synergy
C_BACKBONE  = RGBColor(0xF5, 0x8E, 0x1E)   # orange — Shared backbone
C_NOISE     = RGBColor(0x88, 0x88, 0x88)   # grey   — Noise
C_PANEL     = RGBColor(0x0D, 0x0D, 0x1E)   # darker panel fill
C_BORDER    = RGBColor(0x33, 0x33, 0x55)   # border
C_TEXT      = RGBColor(0xE8, 0xE8, 0xF0)   # body text
C_WHITE     = RGBColor(0xFF, 0xFF, 0xFF)
C_MATH_BG   = RGBColor(0x10, 0x10, 0x28)

SLIDE_W = Inches(13.33)
SLIDE_H = Inches(7.5)


# ── helpers ───────────────────────────────────────────────────────────────────

def _new_prs() -> Presentation:
    prs = Presentation()
    prs.slide_width  = SLIDE_W
    prs.slide_height = SLIDE_H
    return prs


def _blank_slide(prs: Presentation):
    layout = prs.slide_layouts[6]   # completely blank layout
    return prs.slides.add_slide(layout)


def _fill_bg(slide, color: RGBColor):
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = color


def _add_rect(slide, l, t, w, h, fill: RGBColor, border: RGBColor = None,
              border_pt: float = 0.0, radius: float = 0.0):
    shape = slide.shapes.add_shape(
        1,  # MSO_SHAPE_TYPE.RECTANGLE
        Inches(l), Inches(t), Inches(w), Inches(h),
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    if border and border_pt > 0:
        shape.line.color.rgb = border
        shape.line.width = Pt(border_pt)
    else:
        shape.line.fill.background()
    return shape


def _add_textbox(slide, l, t, w, h, text: str, font_size: float,
                 bold: bool = False, color: RGBColor = None, align=PP_ALIGN.LEFT,
                 italic: bool = False, wrap: bool = True):
    txBox = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    txBox.word_wrap = wrap
    tf = txBox.text_frame
    tf.word_wrap = wrap
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.italic = italic
    if color:
        run.font.color.rgb = color
    return txBox


def _add_line(slide, x1, y1, x2, y2, color: RGBColor, width_pt: float = 1.5):
    from pptx.util import Inches as I
    connector = slide.shapes.add_connector(
        1,  # MSO_CONNECTOR_TYPE.STRAIGHT
        I(x1), I(y1), I(x2), I(y2),
    )
    connector.line.color.rgb = color
    connector.line.width = Pt(width_pt)
    return connector


# ── Slide 1: Schematic ───────────────────────────────────────────────────────

def build_slide1(prs: Presentation):
    slide = _blank_slide(prs)
    _fill_bg(slide, C_BG)

    # Title bar
    _add_rect(slide, 0, 0, 13.33, 0.72, fill=RGBColor(0x0D, 0x0D, 0x24))
    _add_textbox(slide, 0.18, 0.08, 11, 0.55,
                 "PID-SAR-3: Data Generation Pipeline",
                 font_size=22, bold=True, color=C_TITLE, align=PP_ALIGN.LEFT)
    _add_textbox(slide, 0.18, 0.5, 11, 0.28,
                 "Three-view synthetic benchmark  |  10 PID atom types  |  d=32 dimensional observations",
                 font_size=9, color=C_SUBTITLE, align=PP_ALIGN.LEFT)

    # ── Column headers ─────────────────────────────────────────────────────
    headers = [
        (0.15, "LATENT FACTORS",  C_SUBTITLE),
        (3.55, "PID ATOM TYPES",  C_SUBTITLE),
        (9.2,  "3-VIEW OUTPUTS",  C_SUBTITLE),
    ]
    for lx, txt, col in headers:
        _add_rect(slide, lx, 0.82, 3.1 if lx < 3 else (5.3 if lx < 9 else 3.9), 0.32,
                  fill=C_PANEL, border=C_BORDER, border_pt=0.5)
        _add_textbox(slide, lx + 0.08, 0.85, 3.0 if lx < 9 else 3.8, 0.28,
                     txt, font_size=8, bold=True, color=col, align=PP_ALIGN.LEFT)

    # ── PID atom rows ───────────────────────────────────────────────────────
    # Each row: (label, color, latent_desc, view_pattern, note)
    rows = [
        # label              col       latent               views        note
        ("U1 — Unique v1",   C_UNIQUE, "u ~ N(0, Iₘ)",     "■ · ·",    "only x₁ carries signal"),
        ("U2 — Unique v2",   C_UNIQUE, "u ~ N(0, Iₘ)",     "· ■ ·",    "only x₂ carries signal"),
        ("U3 — Unique v3",   C_UNIQUE, "u ~ N(0, Iₘ)",     "· · ■",    "only x₃ carries signal"),
        ("R12 — Redund. 1↔2",C_REDUND,"r,η₁,η₂~ N(0,Iₘ)", "■ ■ ·",    "ρ-noisy copies in x₁, x₂"),
        ("R13 — Redund. 1↔3",C_REDUND,"r,η₁,η₃~ N(0,Iₘ)", "■ · ■",    "ρ-noisy copies in x₁, x₃"),
        ("R23 — Redund. 2↔3",C_REDUND,"r,η₂,η₃~ N(0,Iₘ)", "· ■ ■",    "ρ-noisy copies in x₂, x₃"),
        ("R123 — Redund. all",C_REDUND,"r,η₁,η₂,η₃~N(0,Iₘ)","■ ■ ■",  "all three views correlated"),
        ("S12→3 — Synergy",  C_SYNERGY,"a,b ~ N(0, Iₘ)",   "● ● ★",    "MLP(a,b)→x₃; de-leaked"),
        ("S13→2 — Synergy",  C_SYNERGY,"a,b ~ N(0, Iₘ)",   "● · ★",    "MLP(a,b)→x₂; de-leaked"),
        ("S23→1 — Synergy",  C_SYNERGY,"a,b ~ N(0, Iₘ)",   "· ● ★",    "MLP(a,b)→x₁; de-leaked"),
    ]

    row_h   = 0.54
    row_top = 1.22

    for i, (label, col, latent, views, note) in enumerate(rows):
        y = row_top + i * row_h
        shade = RGBColor(
            min(0xFF, C_PANEL[0] + (i % 2) * 8),
            min(0xFF, C_PANEL[1] + (i % 2) * 8),
            min(0xFF, C_PANEL[2] + (i % 2) * 8),
        )

        # Latent box
        _add_rect(slide, 0.15, y, 3.25, row_h - 0.06, fill=shade, border=C_BORDER, border_pt=0.4)
        _add_textbox(slide, 0.22, y + 0.04, 3.1, 0.22, latent, font_size=8.5, color=C_TEXT)

        # Atom label box
        _add_rect(slide, 3.55, y, 5.5, row_h - 0.06, fill=col,
                  border=RGBColor(max(0, col[0]-30), max(0, col[1]-30), max(0, col[2]-30)),
                  border_pt=0.6)
        _add_textbox(slide, 3.62, y + 0.04, 3.2, 0.24, label, font_size=9, bold=True, color=C_WHITE)
        _add_textbox(slide, 3.62, y + 0.27, 3.7, 0.22, f"  views: {views}", font_size=8, color=C_WHITE)
        _add_textbox(slide, 6.9,  y + 0.04, 2.0, 0.44, note, font_size=7.5, italic=True, color=C_WHITE)

        # Arrow → outputs
        _add_line(slide, 9.08, y + (row_h - 0.06) / 2, 9.22, y + (row_h - 0.06) / 2,
                  color=col, width_pt=1.2)

        # View presence dot strip
        view_colors = []
        parts = views.split()
        for sym in parts:
            if sym == "■":
                view_colors.append(col)
            elif sym == "●":
                view_colors.append(col)
            elif sym == "★":
                view_colors.append(C_SYNERGY)
            else:
                view_colors.append(C_PANEL)
        for vi, (vsym, vcol) in enumerate(zip(parts, view_colors)):
            vx = 9.25 + vi * 1.28
            _add_rect(slide, vx, y + 0.08, 1.1, row_h - 0.22,
                      fill=vcol if vsym != "·" else shade,
                      border=C_BORDER, border_pt=0.4)
            view_label = ["x₁", "x₂", "x₃"][vi]
            _add_textbox(slide, vx + 0.08, y + 0.10, 0.95, 0.32,
                         view_label if vsym != "·" else "—",
                         font_size=8, bold=(vsym != "·"), color=C_WHITE, align=PP_ALIGN.CENTER)

    # ── View column header labels ──────────────────────────────────────────
    for vi, vl in enumerate(["x₁", "x₂", "x₃"]):
        vx = 9.25 + vi * 1.28
        _add_rect(slide, vx, 0.84, 1.1, 0.30, fill=RGBColor(0x20, 0x20, 0x40),
                  border=C_BORDER, border_pt=0.5)
        _add_textbox(slide, vx, 0.86, 1.1, 0.26, vl,
                     font_size=9, bold=True, color=C_SUBTITLE, align=PP_ALIGN.CENTER)

    # ── Shared backbone + noise footer ────────────────────────────────────
    fy = row_top + len(rows) * row_h + 0.08
    _add_rect(slide, 0.15, fy, 12.7, 0.52, fill=RGBColor(0x15, 0x15, 0x30),
              border=C_BACKBONE, border_pt=1.0)
    _add_textbox(slide, 0.28, fy + 0.04, 6.0, 0.44,
                 "Shared backbone  (if gain γ > 0):   z ~ N(0, Iₘ)  →  x_v += γ·α·P_shared·z     "
                 "(same z and P_shared across all views — creates cross-view correlation not tied to any PID atom)",
                 font_size=8, color=C_BACKBONE)
    _add_textbox(slide, 6.8, fy + 0.04, 5.8, 0.44,
                 "Noise:  ε_v ~ N(0, σ²I_d)  →  x_v += ε_v  independently per view"
                 "          Multi-atom mode: x_v = Σ_{k=1}^{K} x_v^{atom_k}",
                 font_size=8, color=C_NOISE)

    # ── Legend ───────────────────────────────────────────────────────────
    legend_items = [
        (C_UNIQUE,   "Unique"),
        (C_REDUND,   "Redundancy"),
        (C_SYNERGY,  "Synergy"),
        (C_BACKBONE, "Shared backbone"),
        (C_NOISE,    "Noise / zero"),
    ]
    lx = 0.15
    _add_textbox(slide, lx, 0.75, 1.0, 0.18, "Legend:", font_size=7.5, bold=True, color=C_SUBTITLE)
    lx2 = 0.72
    for lcol, ltxt in legend_items:
        _add_rect(slide, lx2, 0.76, 0.15, 0.14, fill=lcol)
        _add_textbox(slide, lx2 + 0.18, 0.74, 1.3, 0.18, ltxt, font_size=7.5, color=C_TEXT)
        lx2 += 1.55


# ── Slide 2: Mathematics ──────────────────────────────────────────────────────

def build_slide2(prs: Presentation):
    slide = _blank_slide(prs)
    _fill_bg(slide, C_BG)

    # Title bar
    _add_rect(slide, 0, 0, 13.33, 0.68, fill=RGBColor(0x0D, 0x0D, 0x24))
    _add_textbox(slide, 0.18, 0.08, 11, 0.52,
                 "PID-SAR-3: Mathematical Formulation",
                 font_size=22, bold=True, color=C_TITLE, align=PP_ALIGN.LEFT)
    _add_textbox(slide, 0.18, 0.48, 11, 0.24,
                 "d=32 obs. dim  |  m=8 latent dim  |  σ=0.02 noise  |  ρ∈{0.8}  |  α~U[0.8,1.2]  "
                 "|  hop∈{1}  |  λ=0.25 de-leakage  |  γ=4.0 backbone gain  |  K=5 atoms (multi-atom mode)",
                 font_size=8, color=C_SUBTITLE, align=PP_ALIGN.LEFT)

    # ── Four equation panels in a 2×2 grid ────────────────────────────────
    panels = [
        # (left, top, width, height, accent_color, title, lines_of_math)
        (
            0.18, 0.82, 6.3, 2.05, C_UNIQUE,
            "① Unique atom  U_v  (v ∈ {1, 2, 3})",
            [
                "u  ~  N(0, I_m)",
                "",
                "x_v  =  α · P_v^{U_v} · u",
                "",
                "x_{v'} = 0   for all v' ≠ v",
                "",
                "• u is a private m-dim latent, visible only in view v",
                "• P_v^{U_v} ∈ ℝ^{d×m} is a fixed column-normalised random projection",
                "• α ~ U[α_min, α_max] scales the signal amplitude",
            ],
        ),
        (
            6.72, 0.82, 6.42, 2.05, C_REDUND,
            "② Redundancy atoms  R_{ij} / R_{123}",
            [
                "r, η_i, η_j  ~  N(0, I_m)  independently",
                "",
                "r_i  =  √ρ · r  +  √(1−ρ) · η_i",
                "r_j  =  √ρ · r  +  √(1−ρ) · η_j",
                "",
                "x_i = α · P_i^{R_{ij}} · r_i ,   x_j = α · P_j^{R_{ij}} · r_j",
                "x_k = 0   (k ≠ i, j)",
                "",
                "• r is the shared factor; ρ controls correlation strength",
                "• R_{123}: same formula extended to all three views",
            ],
        ),
        (
            0.18, 3.05, 6.3, 2.62, C_SYNERGY,
            "③ Directional Synergy  S_{ij→k}",
            [
                "a, b  ~  N(0, I_m)   (independent source latents)",
                "",
                "x_i = α · P_i^{A_{ij}} · a",
                "x_j = α · P_j^{B_{ij}} · b",
                "",
                "s₀  =  f_MLP(a, b ; hop)     [fixed random residual MLP, tanh activations]",
                "",
                "δ  =  a · C_a[hop]  +  b · C_b[hop]      [linear de-leakage component]",
                "     (C_a, C_b fitted by ridge regression to predict s₀ from a, b)",
                "",
                "s  =  s₀  −  λ · δ          [de-leaked synergy signal]",
                "",
                "x_k = α · P_k^{SYN_{ij}} · s",
                "",
                "• x_k depends non-linearly on BOTH x_i and x_j but not individually",
                "• de-leakage removes the component of s₀ predictable from a or b alone",
            ],
        ),
        (
            6.72, 3.05, 6.42, 2.62, C_BACKBONE,
            "④ Shared backbone, noise & composition",
            [
                "Shared backbone  (applied after atom construction):",
                "   z  ~  N(0, I_m)   [per-sample random latent]",
                "   x_v  ←  x_v  +  γ · α · P^{shared}_v · z",
                "   Same z and same P^{shared}_v across all views → cross-view correlation",
                "   P^{shared} is tied (identical) across views when tied_projection=True",
                "",
                "Observation noise (independent per view):",
                "   ε_v  ~  N(0, σ² I_d)",
                "   x_v  ←  x_v  +  ε_v",
                "",
                "Multi-atom composition (K atoms, K ≥ 2):",
                "   x_v^{total}  =  Σ_{k=1}^{K}  x_v^{atom_k}",
                "   Primary atom label retained; K−1 additional atoms sampled w/o replacement",
                "",
                "Fixed projections:  P[view][comp] ∈ ℝ^{d×m}, columns normalised to unit ℓ₂",
                "Synergy MLP:  residual blocks with tanh; weights fixed at generator init",
            ],
        ),
    ]

    for lx, ty, pw, ph, accent, title, math_lines in panels:
        # Panel background
        _add_rect(slide, lx, ty, pw, ph, fill=C_MATH_BG,
                  border=accent, border_pt=1.2)
        # Accent top stripe
        _add_rect(slide, lx, ty, pw, 0.26, fill=accent)
        _add_textbox(slide, lx + 0.12, ty + 0.02, pw - 0.15, 0.24,
                     title, font_size=9, bold=True, color=C_WHITE)

        # Math content
        content_y = ty + 0.30
        content_h = ph - 0.34
        txBox = slide.shapes.add_textbox(
            Inches(lx + 0.12), Inches(content_y),
            Inches(pw - 0.18), Inches(content_h),
        )
        txBox.word_wrap = True
        tf = txBox.text_frame
        tf.word_wrap = True

        for li, line in enumerate(math_lines):
            if li == 0:
                p = tf.paragraphs[0]
            else:
                p = tf.add_paragraph()
            p.space_before = Pt(0)
            p.space_after  = Pt(0)
            run = p.add_run()
            run.text = line
            is_note = line.startswith("•") or line.startswith("   ")
            run.font.size  = Pt(7.5 if is_note else 8.5)
            run.font.bold  = not is_note and bool(line) and not line.startswith(" ")
            run.font.italic = is_note
            run.font.color.rgb = (
                RGBColor(0xBB, 0xBB, 0xCC) if is_note else
                RGBColor(0xF0, 0xF0, 0xFF)
            )
            if not line:
                run.font.size = Pt(3)

    # ── Bottom legend strip ───────────────────────────────────────────────
    _add_rect(slide, 0.18, 5.74, 12.95, 0.30, fill=RGBColor(0x0D, 0x0D, 0x22),
              border=C_BORDER, border_pt=0.4)
    _add_textbox(
        slide, 0.28, 5.76, 12.7, 0.26,
        "Projections:  P_v^{comp} ~ N(0, 1/√d), col-normalised  |  "
        "α ~ U[0.8, 1.2]  |  ρ ∈ {0.2, 0.5, 0.8}  |  hop ∈ {1,2,3,4}  |  "
        "σ = 0.02 (low noise)  |  λ = 0.25 (de-leakage)  |  γ = 4.0 (backbone)  |  "
        "K = 1 (single-atom) or 5 (multi-atom)  |  Balanced over 10 PID atoms",
        font_size=7.5, color=C_SUBTITLE,
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    prs = _new_prs()
    build_slide1(prs)
    build_slide2(prs)
    out = "pid_ssl_data_generation.pptx"
    prs.save(out)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

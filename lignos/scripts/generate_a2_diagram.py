"""Generate A2 backbone architecture diagram (Figure 2 of LIGNOS paper).

Shows the internal structure of the pretrained A2 backbone that all three
LIGNOS specialists inherit and freeze:

  Inputs:
    - v = 0.4·preds_fusion + 0.6·preds_chemprop (cached upstream ensemble, 8-D)
    - Morgan fingerprints (PCA-40)
    - Thermo features (first 5 dims: T, IL concentration, biomass C/H/O)
    - Chemprop D-MPNN features (PCA-40, cached upstream)

  Blocks (total 20,936 trainable params):
    - Thermo gate: Linear(5 → 32) → GELU → Linear(32 → 40) → Sigmoid
      Produces per-Morgan-dim gating vector g ∈ [0, 1]^40 conditioned on process.
    - 8 property main heads (per-property MLP on [gated-Morgan ⊕ thermo[:5]]).
      Each head: Linear(45 → 32) → GELU → Linear(32 → 1). Zero-init last layer.
    - α gates (8 scalars, init at -3 → sigmoid ≈ 0.047).
    - Chemprop projection: Linear(40 → 32) → GELU → Linear(32 → 32). Zero-init.
    - 8 Chemprop heads (per-property MLP on [cp_proj ⊕ thermo[:5]]).
      Each head: Linear(37 → 16) → GELU → Linear(16 → 1). Zero-init.
    - Chemprop gates (8 scalars, init at -5 → sigmoid ≈ 0.007).

  Output: out = v + σ(α) · main_residual + σ(cp_gate) · chemprop_residual (8-D).
  At initialization both residual gates are near zero, so out ≈ v (cached
  upstream ensemble) — safe starting point for finetuning.

Style matches scripts/generate_lignos_diagram.py.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from pathlib import Path


def draw_box(ax, xy, w, h, text, color="#E8F4FD", edge_color="#2196F3",
             fontsize=9, fontweight="normal", text_color="black", alpha=1.0,
             subtext=None, subtext_size=7.5):
    box = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.08",
                         facecolor=color, edgecolor=edge_color,
                         linewidth=1.5, alpha=alpha)
    ax.add_patch(box)
    cx, cy = xy[0] + w / 2, xy[1] + h / 2
    if subtext:
        ax.text(cx, cy + 0.15, text, ha="center", va="center",
                fontsize=fontsize, fontweight=fontweight, color=text_color)
        ax.text(cx, cy - 0.22, subtext, ha="center", va="center",
                fontsize=subtext_size, color="#555555", style="italic")
    else:
        ax.text(cx, cy, text, ha="center", va="center",
                fontsize=fontsize, fontweight=fontweight, color=text_color)


def draw_arrow(ax, start, end, color="#666666", lw=1.2, style="->",
               connectionstyle=None, alpha=1.0):
    if connectionstyle:
        arrow = FancyArrowPatch(start, end, arrowstyle=style,
                                connectionstyle=connectionstyle,
                                color=color, lw=lw, mutation_scale=12,
                                alpha=alpha)
    else:
        arrow = FancyArrowPatch(start, end, arrowstyle=style,
                                color=color, lw=lw, mutation_scale=12,
                                alpha=alpha)
    ax.add_patch(arrow)


def draw_path_arrow(ax, points, color="#666666", lw=1.2, style="->", alpha=1.0):
    """Rectangular multi-segment route with arrowhead on the final segment.

    Intermediate segments render as plain lines (arrowstyle='-'); the last
    segment carries the arrowhead. Use to route around boxes instead of
    drawing arcs that cross them."""
    if len(points) < 2:
        return
    for i in range(len(points) - 1):
        seg_style = style if i == len(points) - 2 else "-"
        arrow = FancyArrowPatch(points[i], points[i + 1], arrowstyle=seg_style,
                                color=color, lw=lw, mutation_scale=12,
                                alpha=alpha)
        ax.add_patch(arrow)


def main():
    fig, ax = plt.subplots(1, 1, figsize=(18, 11))
    ax.set_xlim(-0.5, 18)
    ax.set_ylim(-0.5, 11)
    ax.axis("off")

    # Title
    ax.text(9, 10.5,
            "A2 backbone — multi-task property prediction head "
            "with zero-init gated residuals",
            ha="center", va="center", fontsize=15, fontweight="bold",
            color="#1a1a1a")
    ax.text(9, 10.05,
            "20,936 trainable parameters  ·  pretrained on ILThermo "
            "(60,000 measurements, 7 thermodynamic targets)  ·  frozen in LIGNOS",
            ha="center", va="center", fontsize=9, color="#555555", style="italic")

    # ═══════════════════════════════════════════════════════════════
    # ROW 1: CACHED UPSTREAM INPUTS (top)
    # ═══════════════════════════════════════════════════════════════
    y_in = 8.6

    # Upstream encoder note (far left, spanning)
    upstream_box = FancyBboxPatch((0.2, y_in - 0.4), 4.0, 1.5,
                                  boxstyle="round,pad=0.12",
                                  facecolor="#F5F5F5", edgecolor="#9E9E9E",
                                  linewidth=1.3, linestyle=(0, (3, 2)),
                                  alpha=0.6)
    ax.add_patch(upstream_box)
    ax.text(2.2, y_in + 0.8, "Upstream pretrained encoders",
            ha="center", fontsize=10, fontweight="bold", color="#616161",
            style="italic")
    ax.text(2.2, y_in + 0.4, "V4 fusion net  ·  Chemprop D-MPNN",
            ha="center", fontsize=9, color="#616161")
    ax.text(2.2, y_in - 0.0,
            "outputs cached · not retrained in A2 or LIGNOS",
            ha="center", fontsize=7.5, style="italic", color="#757575")

    # Inputs to A2Head
    draw_box(ax, (5.0, y_in), 2.4, 0.9,
             "v = 0.4·p_fus + 0.6·p_cp",
             color="#E3F2FD", edge_color="#1976D2",
             fontsize=10, fontweight="bold",
             subtext="dual-path v4 base (8-D)")
    draw_box(ax, (7.8, y_in), 2.4, 0.9,
             "Morgan PCA-40",
             color="#E8F5E9", edge_color="#2E7D32",
             fontsize=10, fontweight="bold",
             subtext="i ∈ ℝ⁴⁰")
    draw_box(ax, (10.6, y_in), 2.4, 0.9,
             "Thermo features",
             color="#FFF3E0", edge_color="#F57C00",
             fontsize=10, fontweight="bold",
             subtext="t ∈ ℝ⁵ (T, x₁, bio C/H/O)")
    draw_box(ax, (13.4, y_in), 2.4, 0.9,
             "Chemprop PCA-40",
             color="#E1F5FE", edge_color="#0288D1",
             fontsize=10, fontweight="bold",
             subtext="cp ∈ ℝ⁴⁰")

    # Arrow: upstream → v (direct) and upstream → Chemprop PCA (routed above
    # the input row so it doesn't cross v / Morgan / Thermo boxes)
    draw_arrow(ax, (4.2, y_in + 0.45), (5.0, y_in + 0.45),
               color="#9E9E9E", lw=1.5)
    draw_path_arrow(ax,
                    [(2.2, 9.7), (2.2, 9.85), (14.6, 9.85), (14.6, 9.5)],
                    color="#9E9E9E", lw=1.0, alpha=0.5)

    # ═══════════════════════════════════════════════════════════════
    # ROW 2: TWO PARALLEL RESIDUAL PATHS
    # ═══════════════════════════════════════════════════════════════

    # ── PATH A (left): main residual (Morgan + thermo)
    ax.text(4.5, 7.1, "Path A — gated Morgan residual",
            ha="center", fontsize=11, fontweight="bold", color="#2E7D32")

    # Thermo gate
    draw_box(ax, (1.0, 5.5), 3.0, 1.2,
             "Thermo gate\nLinear(5→32) → GELU → Linear(32→40) → σ",
             color="#C8E6C9", edge_color="#2E7D32", fontsize=9,
             fontweight="bold",
             subtext="produces g ∈ [0,1]⁴⁰")
    # Gated Morgan
    draw_box(ax, (5.0, 5.5), 3.0, 1.2,
             "Gated Morgan\n g ⊙ i  ∈ ℝ⁴⁰",
             color="#E8F5E9", edge_color="#2E7D32", fontsize=9,
             fontweight="bold",
             subtext="elementwise modulation")
    # Property heads (main)
    draw_box(ax, (1.0, 3.5), 7.0, 1.4,
             "8 per-property main heads   ·   Linear(45→32) → GELU → Linear(32→1)",
             color="#DCEDC8", edge_color="#558B2F", fontsize=10,
             fontweight="bold",
             subtext="input: [gated-Morgan ⊕ thermo]  ·  zero-init last layer  ·  8 scalars out")
    # Alpha gates
    draw_box(ax, (1.0, 2.3), 7.0, 0.9,
             "× σ(α_p)  per property   (init α = −3 ⇒ σ ≈ 0.047)",
             color="#A5D6A7", edge_color="#388E3C", fontsize=9.5,
             fontweight="bold",
             subtext="8 trainable scalar gates — blend residual with v")

    # Path A arrows
    # thermo → gate: route along corridor y=7.5 (below inputs, above path boxes)
    draw_path_arrow(ax,
                    [(11.8, y_in), (11.8, 7.5), (2.5, 7.5), (2.5, 6.7)],
                    color="#F57C00", lw=1.5)
    draw_arrow(ax, (8.8, y_in), (6.5, 6.7), color="#2E7D32", lw=1.5)  # Morgan → gated Morgan
    draw_arrow(ax, (2.5, 5.5), (2.5, 4.9), color="#2E7D32", lw=1.5)  # gate → heads
    draw_arrow(ax, (6.5, 5.5), (6.5, 4.9), color="#2E7D32", lw=1.5)  # gated Morgan → heads
    draw_arrow(ax, (4.5, 3.5), (4.5, 3.2), color="#2E7D32", lw=1.5)  # heads → alpha

    # ── PATH B (right): Chemprop residual
    ax.text(13.5, 7.1, "Path B — Chemprop residual",
            ha="center", fontsize=11, fontweight="bold", color="#0288D1")

    # cp_proj
    draw_box(ax, (10.0, 5.5), 3.0, 1.2,
             "cp_proj\nLinear(40→32) → GELU → Linear(32→32)",
             color="#B3E5FC", edge_color="#0288D1", fontsize=9,
             fontweight="bold",
             subtext="zero-init final layer")
    # Concat with thermo
    draw_box(ax, (13.5, 5.5), 3.0, 1.2,
             "[cp_proj ⊕ thermo]",
             color="#E1F5FE", edge_color="#0288D1", fontsize=10,
             fontweight="bold",
             subtext="ℝ³⁷")
    # CP heads
    draw_box(ax, (10.0, 3.5), 6.5, 1.4,
             "8 Chemprop heads   ·   Linear(37→16) → GELU → Linear(16→1)",
             color="#B3E5FC", edge_color="#0288D1", fontsize=10,
             fontweight="bold",
             subtext="zero-init last layer  ·  8 scalars out")
    # CP gates
    draw_box(ax, (10.0, 2.3), 6.5, 0.9,
             "× σ(cp_gate_p)  per property   (init −5 ⇒ σ ≈ 0.007)",
             color="#81D4FA", edge_color="#0277BD", fontsize=9.5,
             fontweight="bold",
             subtext="8 trainable scalar gates — near-zero at init")

    # Path B arrows
    draw_arrow(ax, (14.6, y_in), (11.5, 6.7), color="#0288D1", lw=1.5)
    draw_arrow(ax, (11.8, y_in), (15.0, 6.7), color="#F57C00", lw=1.2,
               alpha=0.8)  # thermo → concat
    draw_arrow(ax, (13.0, 6.1), (13.5, 6.1), color="#0288D1", lw=1.5)  # cp_proj → concat
    draw_arrow(ax, (14.75, 5.5), (13.25, 4.9), color="#0288D1", lw=1.5)  # concat → heads
    draw_arrow(ax, (13.25, 3.5), (13.25, 3.2), color="#0288D1", lw=1.5)  # heads → gate

    # ═══════════════════════════════════════════════════════════════
    # ROW 3: OUTPUT COMPOSITION
    # ═══════════════════════════════════════════════════════════════

    # Summation block
    draw_box(ax, (4.5, 0.7), 9.0, 1.3,
             "out = v + σ(α) · residual_main + σ(cp_gate) · residual_cp   ∈ ℝ⁸",
             color="#FFF9C4", edge_color="#F57F17", fontsize=12,
             fontweight="bold",
             subtext="8 properties: γ₁, γ₂, H_E, G_E, G_mix, H_vap, density, lignin",
             subtext_size=9)

    # Arrows from both paths + v into output
    draw_arrow(ax, (4.5, 2.3), (7.5, 2.0), color="#2E7D32", lw=1.8)
    draw_arrow(ax, (13.25, 2.3), (10.5, 2.0), color="#0288D1", lw=1.8)
    # v skip: route down the clear corridor at x=9 (between Path A heads
    # ending at x=8 and Path B heads starting at x=10) so the arrow never
    # crosses a box.
    draw_path_arrow(ax,
                    [(6.2, y_in), (6.2, 7.85), (9.0, 7.85), (9.0, 2.0)],
                    color="#1976D2", lw=1.8)
    ax.text(9.2, 5.2, "v (skip)", ha="left", fontsize=8.5,
            color="#1976D2", style="italic", fontweight="bold")

    # ═══════════════════════════════════════════════════════════════
    # FOOTNOTES (bottom)
    # ═══════════════════════════════════════════════════════════════
    ax.text(9, 0.15,
            "At initialization both α and cp_gate are near zero ⇒ out ≈ v (v4 ensemble baseline).  "
            "A2Head learns property-specific corrections on top of the frozen upstream features.",
            ha="center", fontsize=9, color="#424242", style="italic")

    plt.tight_layout()
    out_dir = Path("/work/nvme/bgte/kahmed2/Dataset_Chemistry/"
                   "paper/j_cheminform/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / "fig2_a2_backbone.pdf"
    png = out_dir / "fig2_a2_backbone.png"
    plt.savefig(pdf, dpi=200, bbox_inches="tight")
    plt.savefig(png, dpi=200, bbox_inches="tight")
    print(f"Saved: {pdf}")
    print(f"Saved: {png}")


if __name__ == "__main__":
    main()

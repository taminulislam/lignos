"""Generate LIGNOS architecture diagram for Journal of Cheminformatics paper.

Shows the complete LIGNOS pipeline:
  1. Multi-modal input extraction (Morgan, Chemprop, DFT surface, ViT frames,
     COSMO-SAC sigma-profile, thermo, physchem)
  2. Three architecturally-distinct specialists (A, B, C) with per-specialist
     aleatoric logvar heads
  3. Scalar Bayesian model averaging (BMA) router (24 params)
  4. Stage-1 core-7 output (with confidence gate)
  5. Stage-2 deep lignin head with cross-task feature distillation
     (the model's own predicted core-7 mu fed as input)
  6. Two output arrows: core-7 predictions + lignin prediction

Style mirrors scripts/generate_architecture_diagram.py for visual consistency.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from pathlib import Path


def draw_box(ax, xy, w, h, text, color="#E8F4FD", edge_color="#2196F3",
             fontsize=9, fontweight="normal", text_color="black", alpha=1.0,
             subtext=None, subtext_size=6.8):
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


def main():
    fig, ax = plt.subplots(1, 1, figsize=(26, 13.5))
    ax.set_xlim(-1, 27)
    ax.set_ylim(-0.5, 13.5)
    ax.axis("off")

    # Title
    ax.text(13, 12.9,
            "LIGNOS: Calibrated Multi-Modal Ensemble with Cross-Task Feature Distillation",
            ha="center", va="center", fontsize=16, fontweight="bold",
            color="#1a1a1a")
    ax.text(13, 12.4,
            "Three architecturally-distinct specialists · Scalar BMA router (24 params) · "
            "Stage-2 lignin head conditioned on predicted core-7 μ (detached)",
            ha="center", va="center", fontsize=9, color="#555555", style="italic")

    # ═══════════════════════════════════════════════════════════════
    # COLUMN 1: INPUTS
    # ═══════════════════════════════════════════════════════════════
    ax.text(1.2, 11.6, "Inputs", ha="center", fontsize=11,
            fontweight="bold", color="#D32F2F")

    # SMILES
    draw_box(ax, (0, 9.8), 2.4, 1.3, "IL SMILES",
             color="#FFEBEE", edge_color="#E53935",
             fontsize=9.5, fontweight="bold",
             subtext="cation · anion")
    # Process features
    draw_box(ax, (0, 7.9), 2.4, 1.3, "Process features",
             color="#FFF3E0", edge_color="#FB8C00",
             fontsize=9.5, fontweight="bold",
             subtext="T, IL conc, biomass C/H/O")
    # Physchem
    draw_box(ax, (0, 6.0), 2.4, 1.3, "Physchem table",
             color="#F3E5F5", edge_color="#8E24AA",
             fontsize=9.5, fontweight="bold",
             subtext="12-D (46% imputed)")
    # DFT geometry
    draw_box(ax, (0, 3.5), 2.4, 1.8, "DFT geometry\n(Psi4, B3LYP)",
             color="#E8F5E9", edge_color="#2E7D32",
             fontsize=9.5, fontweight="bold",
             subtext="468 cation·anion pairs")

    # ═══════════════════════════════════════════════════════════════
    # COLUMN 2: FEATURE EXTRACTORS
    # ═══════════════════════════════════════════════════════════════
    ax.text(4.4, 11.6, "Feature extractors", ha="center", fontsize=11,
            fontweight="bold", color="#1565C0")

    # Morgan
    draw_box(ax, (3.2, 10.3), 2.4, 0.8, "Morgan FP (PCA-40)",
             color="#E3F2FD", edge_color="#1976D2", fontsize=8.5)
    # Chemprop
    draw_box(ax, (3.2, 9.3), 2.4, 0.8, "Chemprop D-MPNN (40)",
             color="#E3F2FD", edge_color="#1976D2", fontsize=8.5)
    # Thermo
    draw_box(ax, (3.2, 8.3), 2.4, 0.8, "Thermo feat (5)",
             color="#FFF8E1", edge_color="#FB8C00", fontsize=8.5)
    # Physchem feat
    draw_box(ax, (3.2, 7.3), 2.4, 0.8, "Physchem (12) + mask",
             color="#F3E5F5", edge_color="#8E24AA", fontsize=8.5)
    # PointNet
    draw_box(ax, (3.2, 5.6), 2.4, 1.0, "PointNet\n(COSMO surface)",
             color="#E8F5E9", edge_color="#2E7D32",
             fontsize=8.5, fontweight="bold",
             subtext="256-D (91% coverage)")
    # ViT
    draw_box(ax, (3.2, 4.3), 2.4, 1.0, "ViT-B/16\n(rendered frames)",
             color="#E0F7FA", edge_color="#00838F",
             fontsize=8.5, fontweight="bold",
             subtext="192-D (100% coverage)")
    # COSMO-SAC σ-profile
    draw_box(ax, (3.2, 2.9), 2.4, 1.1, "COSMO-SAC σ-profile\nmoments",
             color="#FFF3E0", edge_color="#E64A19",
             fontsize=8.5, fontweight="bold",
             subtext="20-D (97.9% coverage)")

    # Arrows input → extractors
    draw_arrow(ax, (2.4, 10.6), (3.2, 10.6), color="#E53935", lw=1.3)  # SMILES→Morgan
    draw_arrow(ax, (2.4, 10.4), (3.2, 9.7), color="#E53935", lw=1.1, alpha=0.8)  # SMILES→Chemprop
    draw_arrow(ax, (2.4, 8.5), (3.2, 8.7), color="#FB8C00", lw=1.3)  # Process→Thermo
    draw_arrow(ax, (2.4, 6.6), (3.2, 7.7), color="#8E24AA", lw=1.3)  # Physchem
    # DFT → three extractors
    draw_arrow(ax, (2.4, 4.7), (3.2, 6.1), color="#2E7D32", lw=1.3)
    draw_arrow(ax, (2.4, 4.4), (3.2, 4.8), color="#2E7D32", lw=1.3)
    draw_arrow(ax, (2.4, 4.0), (3.2, 3.4), color="#2E7D32", lw=1.3)

    # ═══════════════════════════════════════════════════════════════
    # COLUMN 3: THREE SPECIALISTS
    # ═══════════════════════════════════════════════════════════════
    # ── Shared-feature bus: one vertical collector line at x=6.5 catches the
    # four shared-feature streams; three labelled branches fan out to each
    # specialist on the right. This replaces the previous 12-arrow tangle.
    bus_x = 6.5
    # Vertical spine
    ax.plot([bus_x, bus_x], [7.6, 10.8], color="#607D8B", lw=2.2, alpha=0.85,
            solid_capstyle="round")
    # Feature inlets into the bus (4 shared features)
    for src_y in (10.7, 9.7, 8.7, 7.7):
        draw_arrow(ax, (5.6, src_y), (bus_x, src_y),
                   color="#666", lw=1.0, alpha=0.7)
    # Bus label
    ax.text(bus_x, 11.0,
            "A2 backbone\nfeatures",
            ha="center", va="center", fontsize=7.5, fontweight="bold",
            color="#37474F", style="italic")

    # Three specialists now sit further right at x=8.4
    spec_x = 8.4
    spec_w = 5.0

    ax.text(spec_x + spec_w / 2, 11.6,
            "Three specialists (GraphSpec · SurfSpec · SigmaSpec)",
            ha="center", fontsize=11, fontweight="bold", color="#BF360C")

    # ── Shared frozen A2 backbone: dashed enclosure around all 3 specialists
    backbone_enclosure = FancyBboxPatch(
        (spec_x - 0.25, 2.55), spec_w + 0.5, 7.95, boxstyle="round,pad=0.2",
        facecolor="#ECEFF1", edgecolor="#546E7A",
        linewidth=1.8, linestyle=(0, (5, 3)), alpha=0.35)
    ax.add_patch(backbone_enclosure)
    ax.text(spec_x + spec_w / 2, 10.7,
            "Shared frozen A2 backbone (940K params)",
            ha="center", va="center", fontsize=8, fontweight="bold",
            color="#37474F", style="italic")

    # GraphSpec (A)
    draw_box(ax, (spec_x, 8.9), spec_w, 1.4,
             "GraphSpec  (Specialist A)",
             color="#FFF8E1", edge_color="#F9A825",
             fontsize=12.5, fontweight="bold",
             subtext="A2 + aleatoric logvar head (bias +2, clamp [−3,+3])",
             subtext_size=9)
    # SurfSpec (B)
    draw_box(ax, (spec_x, 5.0), spec_w, 1.4,
             "SurfSpec  (Specialist B)",
             color="#E1F5FE", edge_color="#0288D1",
             fontsize=12.5, fontweight="bold",
             subtext="A2 + zero-init gated residual: PointNet(256) ⊕ ViT(192)",
             subtext_size=9)
    # SigmaSpec (C)
    draw_box(ax, (spec_x, 2.9), spec_w, 1.4,
             "SigmaSpec  (Specialist C)",
             color="#FFF3E0", edge_color="#E64A19",
             fontsize=12.5, fontweight="bold",
             subtext="A2 + zero-init gated residual: σ-profile moments (20)",
             subtext_size=9)

    # Output equation tags per specialist — anchored to TOP-LEFT corner of
    # each specialist box, left-aligned. Placed just inside the top edge.
    tag_x = spec_x + 0.25
    for ypos, lab in [(10.12, "(μ_A, log σ²_A) ∈ ℝ⁸"),
                       (6.22, "(μ_B, log σ²_B) ∈ ℝ⁸"),
                       (4.12, "(μ_C, log σ²_C) ∈ ℝ⁸")]:
        ax.text(tag_x, ypos, lab, ha="left", va="center",
                fontsize=9.5, color="#424242", style="italic")

    # Bus → each specialist (3 clean branches, not 12)
    # Branch from bus bottom to GraphSpec at y=9.6
    draw_arrow(ax, (bus_x, 9.6), (spec_x, 9.6), color="#546E7A", lw=1.8)
    # Branch from bus to SurfSpec
    draw_arrow(ax, (bus_x, 8.0), (spec_x, 5.7), color="#546E7A", lw=1.8,
               connectionstyle="arc3,rad=-0.15")
    # Branch from bus to SigmaSpec
    draw_arrow(ax, (bus_x, 7.7), (spec_x, 3.6), color="#546E7A", lw=1.8,
               connectionstyle="arc3,rad=-0.18")

    # Modality-specific arrows: surface+ViT → SurfSpec (B only); σ-profile → SigmaSpec (C only)
    draw_arrow(ax, (5.6, 6.1), (spec_x, 5.9), color="#2E7D32", lw=1.8)   # surface
    draw_arrow(ax, (5.6, 4.8), (spec_x, 5.3), color="#00838F", lw=1.8)   # ViT
    draw_arrow(ax, (5.6, 3.4), (spec_x, 3.3), color="#E64A19", lw=1.8)   # σ-profile

    # ═══════════════════════════════════════════════════════════════
    # COLUMN 4: BMA ROUTER
    # ═══════════════════════════════════════════════════════════════
    router_x = 14.8
    router_w = 3.0
    router_cx = router_x + router_w / 2

    ax.text(router_cx, 11.6, "Scalar BMA router", ha="center", fontsize=11,
            fontweight="bold", color="#6A1B9A")

    # Router box spanning
    router_box = FancyBboxPatch((router_x, 4.2), router_w, 5.5,
                                 boxstyle="round,pad=0.15",
                                 facecolor="#F3E5F5", edgecolor="#7B1FA2",
                                 linewidth=2.0, alpha=0.4)
    ax.add_patch(router_box)

    draw_box(ax, (router_x + 0.2, 8.7), router_w - 0.4, 0.85,
             "Anchor log(1/σ²_k)",
             color="#CE93D8", edge_color="#7B1FA2", fontsize=10.5,
             fontweight="bold",
             subtext="per-specialist precision",
             subtext_size=8.5)
    draw_box(ax, (router_x + 0.2, 7.55), router_w - 0.4, 0.85,
             "+ scalar bias b ∈ ℝ^{K×P}",
             color="#E1BEE7", edge_color="#7B1FA2", fontsize=10.5,
             fontweight="bold",
             subtext="24 parameters (K=3, P=8)",
             subtext_size=8.5)
    draw_box(ax, (router_x + 0.2, 6.4), router_w - 0.4, 0.85,
             "softmax over K",
             color="#E1BEE7", edge_color="#7B1FA2", fontsize=10.5,
             fontweight="bold",
             subtext="→ weights w_k(p)",
             subtext_size=8.5)
    draw_box(ax, (router_x + 0.2, 5.0), router_w - 0.4, 1.2,
             "μ̂ = Σ w_k μ_k\nσ̂² = (Σ exp(−log σ²_k))⁻¹",
             color="#F8BBD0", edge_color="#7B1FA2", fontsize=10,
             fontweight="bold",
             subtext="precision-weighted BMA fusion",
             subtext_size=8.5)
    # label
    ax.text(router_cx, 4.4, "Beats 21k-param MLP on all metrics",
            ha="center", fontsize=9, color="#4A148C", style="italic")

    # Arrows from specialists into router
    spec_right = spec_x + spec_w
    draw_arrow(ax, (spec_right, 9.6), (router_x, 9.2), color="#F9A825", lw=1.6)
    draw_arrow(ax, (spec_right, 5.7), (router_x, 8.1), color="#0288D1", lw=1.6)
    draw_arrow(ax, (spec_right, 3.6), (router_x, 6.4), color="#E64A19", lw=1.6)

    # ═══════════════════════════════════════════════════════════════
    # COLUMN 5: OUTPUTS
    # ═══════════════════════════════════════════════════════════════
    out_x = 19.6
    out_w = 4.8
    out_cx = out_x + out_w / 2
    router_right = router_x + router_w

    ax.text(out_cx, 11.6, "Outputs", ha="center", fontsize=11,
            fontweight="bold", color="#00695C")

    # Stage 1 output: core-7
    draw_box(ax, (out_x, 9.0), out_w, 1.7,
             "Stage 1 · Core-7 thermo",
             color="#E0F2F1", edge_color="#00695C",
             fontsize=13, fontweight="bold",
             subtext="R² = 0.834 (all)  ·  R² = 0.935 (gated@50%)",
             subtext_size=9.5)

    # Confidence gate
    draw_box(ax, (out_x, 7.4), out_w, 1.3,
             "Confidence gate",
             color="#B2DFDB", edge_color="#00695C",
             fontsize=11.5, fontweight="bold",
             subtext="σ²_total = aleatoric + epistemic  →  top-50% retention",
             subtext_size=8.8)

    # Stage 2 header
    ax.text(out_cx, 6.65, "Stage 2 · cross-task feature distillation",
            ha="center", fontsize=11, fontweight="bold", color="#006064")

    # Stage-2 input composition
    s2_x = out_x - 0.8
    s2_w = out_w + 1.0
    draw_box(ax, (s2_x, 4.7), s2_w, 1.7,
             "deep_lignin MLP (128 → 64 → 1)",
             color="#B2EBF2", edge_color="#006064",
             fontsize=12, fontweight="bold",
             subtext="input: gated(chemprop) ⊕ thermo[:5] ⊕ physchem ⊕ has_phys "
                     "⊕ μ̂_core[:7].detach()",
             subtext_size=9)

    # Stage 2 output
    draw_box(ax, (out_x, 2.7), out_w, 1.6,
             "Stage 2 · Lignin yield",
             color="#E0F7FA", edge_color="#00838F",
             fontsize=13, fontweight="bold",
             subtext="R² = 0.750 ± 0.037  (vs. Baran GB 0.44, Δ = +0.31)",
             subtext_size=9.5)

    # Arrows from router to Stage 1 / Stage 2
    draw_arrow(ax, (router_right, 8.0), (out_x, 9.6), color="#7B1FA2", lw=2.0)  # → Stage 1
    draw_arrow(ax, (router_right, 6.6), (out_x, 8.1), color="#7B1FA2", lw=1.5,
               alpha=0.85)  # → confidence gate
    draw_arrow(ax, (router_right, 6.2), (s2_x, 5.6), color="#7B1FA2", lw=1.8)  # → Stage 2 MLP

    # μ̂_core feeding back into Stage 2 (the KEY distillation arrow)
    # Curve bulges LEFT (rad=-0.35) so the bulge sits at x < fb_x; the label
    # floats in the right margin (x > out_x + out_w), outside all boxes.
    fb_x = out_cx
    draw_arrow(ax, (fb_x, 9.1), (fb_x, 6.4), color="#D81B60", lw=2.4,
               connectionstyle="arc3,rad=-0.35")
    label_x = out_x + out_w + 0.35
    ax.text(label_x, 7.55, "μ̂_core[:7]\n(detached)",
            ha="left", va="center", fontsize=10, fontweight="bold",
            color="#D81B60")

    # Stage 2 residual arrow
    draw_arrow(ax, (fb_x, 4.8), (fb_x, 4.3), color="#00838F", lw=1.8)

    # Alpha gate label
    ax.text(fb_x, 4.55, "σ(α_lignin) · residual",
            ha="center", fontsize=7, color="#00838F", style="italic")

    # ═══════════════════════════════════════════════════════════════
    # A2 BACKBONE SUMMARY BOX
    # Small summary panel placed directly under the backbone enclosure,
    # using matching dashed gray styling so the two read as the same
    # component. Full schematic lives in Fig. 2.
    # ═══════════════════════════════════════════════════════════════
    a2_x, a2_y = 8.15, 0.98
    a2_w, a2_h = 5.5, 1.50
    a2_box = FancyBboxPatch((a2_x, a2_y), a2_w, a2_h,
                            boxstyle="round,pad=0.12",
                            facecolor="#ECEFF1", edgecolor="#546E7A",
                            linewidth=1.6, linestyle=(0, (5, 3)), alpha=0.7)
    ax.add_patch(a2_box)
    # Title
    ax.text(a2_x + a2_w / 2, a2_y + a2_h - 0.22,
            "A2 backbone — summary  (see Fig. 2 for full schematic)",
            ha="center", va="center", fontsize=9, fontweight="bold",
            color="#37474F")
    # Body bullets
    a2_lines = [
        "• 20,936 trainable params · pretrained on ILThermo (60K rows, 7 thermo targets)",
        r"• out $= \mathbf{v} + \sigma(\alpha)\,\mathbf{r}_{\mathrm{main}} + \sigma(\mathrm{cp\_gate})\,\mathbf{r}_{\mathrm{cp}} \in \mathbb{R}^{8}$  ($\mathbf{v}$ = cached $0.4\,p_{\mathrm{fus}}{+}0.6\,p_{\mathrm{cp}}$)",
        "• Frozen in LIGNOS · zero-init residual gates ⇒ model starts at v baseline",
    ]
    for i, line in enumerate(a2_lines):
        ax.text(a2_x + 0.22, a2_y + a2_h - 0.55 - i * 0.32, line,
                ha="left", va="center", fontsize=7.6, color="#37474F")

    # Dashed tick connecting summary box to the backbone enclosure above
    # (both share x=8.15..13.65, so a short vertical tether makes the
    # "this describes the A2 backbone" link explicit).
    ax.plot([a2_x + a2_w / 2, a2_x + a2_w / 2], [a2_y + a2_h, 2.55],
            color="#546E7A", lw=1.0, linestyle=(0, (2, 2)), alpha=0.6)

    # ═══════════════════════════════════════════════════════════════
    # LEGEND FOOTER
    # ═══════════════════════════════════════════════════════════════
    legend_entries = [
        ("#E53935", "SMILES"),
        ("#FB8C00", "Process features"),
        ("#8E24AA", "Physchem"),
        ("#2E7D32", "DFT surface"),
        ("#00838F", "ViT frames"),
        ("#E64A19", "COSMO-SAC"),
        ("#7B1FA2", "BMA router"),
        ("#D81B60", "Cross-task distillation"),
    ]
    legend_x = 0.5
    legend_y = 0.6
    for i, (color, label) in enumerate(legend_entries):
        x0 = legend_x + (i % 4) * 6.5
        y0 = legend_y - (i // 4) * 0.5
        draw_box(ax, (x0, y0), 0.35, 0.28, "", color=color,
                 edge_color=color, fontsize=7)
        ax.text(x0 + 0.5, y0 + 0.14, label, ha="left", va="center",
                fontsize=8.5, color="#333333")

    ax.text(13, -0.3,
            "Dataset: LignoIL unified (200 rows · 52 SMILES · 45 biomasses) · "
            "Sources: ILThermo + Baran 2024 · Code & checkpoints: github.com/<TBD>/lignos",
            ha="center", fontsize=8, color="#888888", style="italic")

    plt.tight_layout()

    # Save to paper figures dir
    out_dir = Path("/work/nvme/bgte/kahmed2/Dataset_Chemistry/"
                   "paper/j_cheminform/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / "fig1_architecture.pdf"
    png = out_dir / "fig1_architecture.png"
    plt.savefig(pdf, dpi=200, bbox_inches="tight")
    plt.savefig(png, dpi=200, bbox_inches="tight")
    print(f"Saved: {pdf}")
    print(f"Saved: {png}")


if __name__ == "__main__":
    main()

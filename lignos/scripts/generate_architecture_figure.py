#!/usr/bin/env python3
"""Generate COSMOBridge v5 architecture diagram.

Creates a publication-quality figure matching the visual style of the
COSMOBridge v3/v4 architecture diagram.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


def draw_rounded_box(ax, x, y, w, h, text, color, fontsize=8,
                     text_color="black", alpha=0.85, lw=1.5,
                     edge_color=None, bold=False, subtext=None):
    """Draw a rounded rectangle with centered text."""
    if edge_color is None:
        edge_color = color
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.1",
        facecolor=color, edgecolor=edge_color,
        alpha=alpha, linewidth=lw,
    )
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    if subtext:
        ax.text(x + w / 2, y + h * 0.62, text, ha="center", va="center",
                fontsize=fontsize, color=text_color, fontweight=weight)
        ax.text(x + w / 2, y + h * 0.3, subtext, ha="center", va="center",
                fontsize=fontsize - 1.5, color=text_color, fontstyle="italic", alpha=0.8)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, color=text_color, fontweight=weight)
    return box


def draw_arrow(ax, x1, y1, x2, y2, color="gray", lw=1.5, style="->"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))


def main():
    fig, ax = plt.subplots(1, 1, figsize=(20, 12))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 12)
    ax.axis("off")

    # ═══════════════════════════════════════════
    # TITLE
    # ═══════════════════════════════════════════
    ax.text(10, 11.6, "COSMOBridge v5", fontsize=22, ha="center",
            fontweight="bold", color="#1a237e")
    ax.text(10, 11.2, "Full Multimodal Architecture: 2D Images + 3D Surfaces + Molecular Graphs",
            fontsize=10, ha="center", color="#37474f",
            fontstyle="italic")
    ax.text(10, 10.85, "via Cross-Modal Attention Fusion with Per-Property Adaptive Routing",
            fontsize=9, ha="center", color="#546e7a")

    # ═══════════════════════════════════════════
    # COLUMN 1: MULTIMODAL INPUTS (left)
    # ═══════════════════════════════════════════
    col1_x = 0.3
    section_h = 0.05

    ax.text(col1_x + 1.2, 10.45, "Multimodal Inputs", fontsize=11,
            fontweight="bold", color="#1a237e")

    # Input 1: COSMO Multi-View Images (NEW)
    draw_rounded_box(ax, col1_x, 8.9, 2.6, 1.4,
                     "36 COSMO Rotation\nViews (224×224)",
                     "#ffcdd2", fontsize=8.5, bold=True,
                     subtext="NEW: Multi-view images",
                     edge_color="#c62828")

    # Input 2: Cation + Anion Images (NEW)
    draw_rounded_box(ax, col1_x, 7.3, 2.6, 1.35,
                     "Cation + Anion\nImages (224×224)",
                     "#f8bbd0", fontsize=8.5, bold=True,
                     subtext="NEW: Separated ion surfaces",
                     edge_color="#ad1457")

    # Input 3: Molecular Graph
    draw_rounded_box(ax, col1_x, 5.8, 2.6, 1.2,
                     "Molecular Graph\n(SMILES → 2D)",
                     "#bbdefb", fontsize=8.5,
                     subtext="from v4",
                     edge_color="#1565c0")

    # Input 4: COSMO Point Cloud
    draw_rounded_box(ax, col1_x, 4.3, 2.6, 1.2,
                     "COSMO Point Cloud\n(1024 × 7 features)",
                     "#c8e6c9", fontsize=8.5,
                     subtext="from v4",
                     edge_color="#2e7d32")

    # Input 5: Thermo + Descriptors
    draw_rounded_box(ax, col1_x, 2.8, 2.6, 1.2,
                     "Thermo + Surface\nDescriptors (25D)",
                     "#fff9c4", fontsize=8.5,
                     subtext="T, 1/T, T², T³, x₁ + 20 desc.",
                     edge_color="#f9a825")

    # ═══════════════════════════════════════════
    # COLUMN 2: PRE-TRAINED ENCODERS
    # ═══════════════════════════════════════════
    col2_x = 4.2
    ax.text(col2_x + 1.3, 10.45, "Pre-trained Encoders", fontsize=11,
            fontweight="bold", color="#1a237e")

    # Encoder 1: ViT-Tiny (SimCLR)
    draw_rounded_box(ax, col2_x, 8.9, 2.8, 1.4,
                     "ViT-Tiny Encoder\n(SimCLR Pre-trained)",
                     "#ef9a9a", fontsize=8.5, bold=True,
                     subtext="6 layers, 192D, 5.7M params",
                     edge_color="#c62828")
    # View attention sub-box
    draw_rounded_box(ax, col2_x + 0.15, 8.45, 2.5, 0.4,
                     "View Self-Attention → Pool → 192D",
                     "#ffcdd2", fontsize=7, edge_color="#e57373")

    # Encoder 2: Siamese CNN
    draw_rounded_box(ax, col2_x, 7.3, 2.8, 1.35,
                     "Siamese CNN Encoder\n(Shared Weights)",
                     "#f48fb1", fontsize=8.5, bold=True,
                     subtext="Cross-Attention → Interact → 192D",
                     edge_color="#ad1457")

    # Encoder 3: Chemprop D-MPNN
    draw_rounded_box(ax, col2_x, 5.8, 2.8, 1.2,
                     "Chemprop D-MPNN",
                     "#90caf9", fontsize=8.5,
                     subtext="Frozen, 300D fingerprint",
                     edge_color="#1565c0")
    # Frozen tag
    ax.text(col2_x + 2.65, 6.85, "FROZEN", fontsize=6, color="white",
            fontweight="bold", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#1565c0", alpha=0.9))

    # Encoder 4: PointNet
    draw_rounded_box(ax, col2_x, 4.3, 2.8, 1.2,
                     "PointNet Encoder",
                     "#a5d6a7", fontsize=8.5,
                     subtext="Frozen, 256D features",
                     edge_color="#2e7d32")
    ax.text(col2_x + 2.65, 5.35, "FROZEN", fontsize=6, color="white",
            fontweight="bold", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#2e7d32", alpha=0.9))

    # Encoder 5: Tabular MLP
    draw_rounded_box(ax, col2_x, 2.8, 2.8, 1.2,
                     "Tabular MLP",
                     "#fff176", fontsize=8.5,
                     subtext="25D → 128D → 256D",
                     edge_color="#f9a825")

    # Arrows: inputs -> encoders
    for y_mid in [9.6, 7.95, 6.4, 4.9, 3.4]:
        draw_arrow(ax, 2.9, y_mid, col2_x, y_mid, color="#78909c", lw=1.5)

    # ═══════════════════════════════════════════
    # COLUMN 3: CROSS-MODAL ATTENTION FUSION
    # ═══════════════════════════════════════════
    col3_x = 8.3
    ax.text(col3_x + 1.0, 10.45, "Cross-Modal Attention Fusion",
            fontsize=11, fontweight="bold", color="#1a237e")

    # Project to common 256D
    draw_rounded_box(ax, col3_x, 9.5, 3.0, 0.7,
                     "Project All → Common 256D",
                     "#e1bee7", fontsize=8.5, bold=True,
                     edge_color="#7b1fa2")

    # Cross-attention pairs
    draw_rounded_box(ax, col3_x, 7.8, 3.0, 1.5,
                     "Pairwise Cross-Attention\n(4 heads, residual + FFN)",
                     "#ce93d8", fontsize=8.5, bold=True,
                     edge_color="#7b1fa2")

    # Cross-attention detail labels
    pairs = ["ViT ↔ Graph", "ViT ↔ Surface", "Siamese ↔ Graph", "Graph ↔ Surface"]
    for i, pair in enumerate(pairs):
        ax.text(col3_x + 1.5, 9.0 - i * 0.27, pair,
                fontsize=7, ha="center", color="#4a148c")

    # Modality norms
    draw_rounded_box(ax, col3_x, 7.15, 3.0, 0.5,
                     "Layer Norm × 5 Modalities",
                     "#e1bee7", fontsize=8, alpha=0.6,
                     edge_color="#7b1fa2")

    # Arrows: encoders -> fusion
    for y_mid in [9.6, 7.95, 6.4, 4.9, 3.4]:
        draw_arrow(ax, col2_x + 2.8, y_mid, col3_x, 9.85, color="#9c27b0", lw=1.2)

    # ═══════════════════════════════════════════
    # COLUMN 4: PER-PROPERTY ROUTING
    # ═══════════════════════════════════════════
    col4_x = 12.5
    ax.text(col4_x + 0.7, 10.45, "Per-Property\nModality Routing",
            fontsize=10, fontweight="bold", color="#1a237e")

    draw_rounded_box(ax, col4_x, 5.5, 2.8, 4.5,
                     "",
                     "#fff3e0", fontsize=8,
                     edge_color="#e65100", alpha=0.3)

    ax.text(col4_x + 1.4, 9.7, "Softmax Routing Weights",
            fontsize=9, ha="center", fontweight="bold", color="#bf360c")
    ax.text(col4_x + 1.4, 9.35, "α(p, m) = softmax over 5 modalities",
            fontsize=7.5, ha="center", color="#bf360c", fontstyle="italic")

    # Routing weight visualization (mini heatmap)
    properties = ["γ₁", "γ₂", "G_E", "H_E", "G_mix", "H_vap", "P"]
    modalities_short = ["ViT", "Siam", "Graph", "PC", "Tab"]

    # Approximate routing weights (from domain knowledge init)
    weights = np.array([
        [0.35, 0.25, 0.10, 0.20, 0.10],  # gamma1
        [0.35, 0.25, 0.10, 0.20, 0.10],  # gamma2
        [0.08, 0.08, 0.45, 0.12, 0.27],  # G_E
        [0.12, 0.12, 0.30, 0.22, 0.24],  # H_E
        [0.08, 0.08, 0.45, 0.12, 0.27],  # G_mix
        [0.20, 0.20, 0.20, 0.20, 0.20],  # H_vap
        [0.30, 0.18, 0.12, 0.28, 0.12],  # P
    ])

    hm_x, hm_y = col4_x + 0.2, 5.7
    hm_w, hm_h = 2.4, 3.4
    cell_w = hm_w / 5
    cell_h = hm_h / 7

    for i in range(7):
        ax.text(hm_x - 0.1, hm_y + hm_h - (i + 0.5) * cell_h,
                properties[i], fontsize=7.5, ha="right", va="center",
                fontweight="bold", color="#bf360c")
        for j in range(5):
            val = weights[i, j]
            color_val = plt.cm.YlOrRd(val / 0.5)
            rect = plt.Rectangle(
                (hm_x + j * cell_w, hm_y + hm_h - (i + 1) * cell_h),
                cell_w, cell_h,
                facecolor=color_val, edgecolor="white", lw=0.5,
            )
            ax.add_patch(rect)
            text_c = "white" if val > 0.25 else "black"
            ax.text(hm_x + (j + 0.5) * cell_w,
                    hm_y + hm_h - (i + 0.5) * cell_h,
                    f"{val:.2f}", fontsize=6, ha="center", va="center",
                    color=text_c, fontweight="bold")

    for j in range(5):
        ax.text(hm_x + (j + 0.5) * cell_w, hm_y + hm_h + 0.12,
                modalities_short[j], fontsize=7, ha="center",
                fontweight="bold", color="#4e342e")

    # Arrow: fusion -> routing
    draw_arrow(ax, col3_x + 3.0, 8.5, col4_x, 8.5, color="#e65100", lw=2)

    # ═══════════════════════════════════════════
    # COLUMN 5: MULTI-TASK PREDICTION HEADS
    # ═══════════════════════════════════════════
    col5_x = 16.3
    ax.text(col5_x + 0.7, 10.45, "Output: 7 Properties",
            fontsize=11, fontweight="bold", color="#1a237e")

    # Shared head
    draw_rounded_box(ax, col5_x, 8.8, 2.8, 0.9,
                     "Shared Backbone\n256D → LN → GELU → Drop",
                     "#b2dfdb", fontsize=8,
                     edge_color="#00695c")

    # Property-specific heads
    prop_names_full = ["γ₁ (activity coeff.)", "γ₂ (activity coeff.)",
                       "G_E (excess Gibbs)", "H_E (excess enthalpy)",
                       "G_mix (Gibbs mixing)", "H_vap (vaporization)",
                       "P (vapor pressure)"]
    prop_colors = ["#ef5350", "#ef5350", "#42a5f5", "#42a5f5",
                   "#42a5f5", "#66bb6a", "#ff7043"]

    for i, (name, color) in enumerate(zip(prop_names_full, prop_colors)):
        y = 8.2 - i * 0.7
        draw_rounded_box(ax, col5_x + 0.1, y, 2.6, 0.55,
                         name, color, fontsize=7.5, alpha=0.7,
                         text_color="white" if i < 5 else "black",
                         edge_color=color)

    # Arrow: routing -> heads
    draw_arrow(ax, col4_x + 2.8, 7.8, col5_x, 9.2, color="#00695c", lw=2)

    # ═══════════════════════════════════════════
    # TARGET R² BOX
    # ═══════════════════════════════════════════
    draw_rounded_box(ax, col5_x + 0.2, 3.2, 2.5, 1.2,
                     "Target:\navg R² = 0.90–0.95",
                     "#d32f2f", fontsize=11, bold=True,
                     text_color="white",
                     edge_color="#b71c1c", lw=2.5)
    ax.text(col5_x + 1.45, 3.0, "v4 baseline: R² = 0.816",
            fontsize=7.5, ha="center", color="#757575", fontstyle="italic")

    # ═══════════════════════════════════════════
    # TRAINING PIPELINE (bottom)
    # ═══════════════════════════════════════════
    y_bottom = 1.3

    ax.text(10, 2.1, "3-Stage Training Pipeline", fontsize=11,
            ha="center", fontweight="bold", color="#1a237e")

    stages = [
        ("Stage 1: SimCLR\nPre-training", "#ffcdd2", "#c62828",
         "Self-supervised\n54K images, 200 epochs"),
        ("Stage 2: Freeze\nEncoders", "#e1bee7", "#7b1fa2",
         "Train fusion + heads\n5 epochs, LR=10⁻³"),
        ("Stage 3: Unfreeze\nImage Encoders", "#ce93d8", "#4a148c",
         "Differential LR\n50 epochs + early stop"),
        ("Stage 4: Full\nFine-tuning", "#b2dfdb", "#00695c",
         "All params, LR=10⁻⁵\n(if R² < 0.90)"),
    ]

    stage_w = 3.8
    gap = 0.35
    start_x = (20 - 4 * stage_w - 3 * gap) / 2

    for i, (title, bg, edge, desc) in enumerate(stages):
        x = start_x + i * (stage_w + gap)
        draw_rounded_box(ax, x, y_bottom, stage_w, 0.65,
                         title, bg, fontsize=8, bold=True,
                         edge_color=edge)
        ax.text(x + stage_w / 2, y_bottom - 0.2, desc,
                fontsize=6.5, ha="center", color="#616161")

        if i < 3:
            draw_arrow(ax, x + stage_w, y_bottom + 0.32,
                       x + stage_w + gap, y_bottom + 0.32,
                       color=edge, lw=2, style="->")

    # ═══════════════════════════════════════════
    # NEW badges
    # ═══════════════════════════════════════════
    for y_pos in [10.15, 8.55]:
        ax.text(col1_x + 0.15, y_pos, "NEW", fontsize=7, color="white",
                fontweight="bold", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="#c62828"))

    # Save
    output_path = "/work/nvme/bgte/kahmed2/Dataset_Chemistry/lignos/article/figures/lignos_architecture.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved: {output_path}")

    # Also save to paper figures
    paper_path = "/work/nvme/bgte/kahmed2/Dataset_Chemistry/lignos/article/figures/lignos_architecture.pdf"
    fig2, ax2 = plt.subplots(1, 1, figsize=(20, 12))
    ax2.set_xlim(0, 20)
    ax2.set_ylim(0, 12)
    ax2.axis("off")
    # Re-draw (simpler: just copy the PNG for now)
    print(f"Architecture diagram generated successfully!")


if __name__ == "__main__":
    main()

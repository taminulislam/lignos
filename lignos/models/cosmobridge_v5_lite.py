"""COSMOBridge v5-Lite: Reduced-capacity model to combat overfitting.

Solution 3: Freeze ALL encoders (ViT, Siamese, Graph, PointNet) and only
train lightweight projections + cross-attention + routing + prediction heads.

Architecture:
    Frozen ViT (192D)      → Linear(192, 128)  ──┐
    Frozen Siamese (192D)  → Linear(192, 128)  ──┤
    Frozen Graph (300D)    → Linear(300, 128)  ──┤→ Cross-Attn → Routing → Heads
    Frozen Surface (256D)  → Linear(256, 128)  ──┤
    Tabular (25D)          → MLP(25, 128)      ──┘

    Trainable: ~200K-500K params (vs 7.6M in full v5)

Supports Solution 6 (knowledge distillation from v4) via distill_loss().
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .multiview_vit import MultiViewViT
from .siamese_encoder import CationAnionSiamese
from .cross_modal_attention import CrossModalFusion


class COSMOBridgeV5Lite(nn.Module):
    """Lightweight COSMOBridge v5 with frozen encoders.

    Parameters
    ----------
    embed_dim : int
        Common projection dimension (default 128, smaller than full v5's 256).
    n_properties : int
        Number of target properties.
    vit_embed_dim : int
        ViT output dimension.
    siamese_embed_dim : int
        Siamese output dimension.
    graph_dim : int
        Frozen graph feature dimension.
    surface_dim : int
        Frozen surface feature dimension.
    thermo_dim : int
        Tabular feature dimension.
    n_cross_attn_heads : int
        Cross-attention heads.
    dropout : float
        Dropout rate.
    use_images : bool
        Whether to include image modalities (set False to match v4 capacity).
    """

    def __init__(
        self,
        embed_dim=128,
        n_properties=7,
        vit_embed_dim=192,
        siamese_embed_dim=192,
        graph_dim=300,
        surface_dim=256,
        thermo_dim=25,
        n_cross_attn_heads=4,
        dropout=0.3,
        use_images=True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_properties = n_properties
        self.use_images = use_images

        # Count active modalities
        self.n_modalities = 3 + (2 if use_images else 0)  # graph + surface + tabular + (vit + siamese)

        # ── Lightweight Projections (TRAINABLE) ──
        if use_images:
            self.vit_proj = nn.Sequential(
                nn.Linear(vit_embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.Dropout(dropout),
            )
            self.siamese_proj = nn.Sequential(
                nn.Linear(siamese_embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.Dropout(dropout),
            )

        self.graph_proj = nn.Sequential(
            nn.Linear(graph_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
        )
        self.surface_proj = nn.Sequential(
            nn.Linear(surface_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
        )
        self.tabular_proj = nn.Sequential(
            nn.Linear(thermo_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── Cross-Modal Attention (TRAINABLE, lightweight) ──
        self.fusion = CrossModalFusion(
            dim=embed_dim,
            n_heads=n_cross_attn_heads,
            n_modalities=self.n_modalities,
            n_properties=n_properties,
            dropout=dropout,
        )

        # ── Prediction Heads (TRAINABLE) ──
        self.shared_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.property_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, 32),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(32, 1),
            )
            for _ in range(n_properties)
        ])

    def forward(self, vit_feat=None, siamese_feat=None,
                graph_feat=None, surface_feat=None, thermo_feat=None):
        """Forward pass with pre-computed frozen features.

        Args:
            vit_feat: (B, vit_dim) - frozen ViT output (or None)
            siamese_feat: (B, siamese_dim) - frozen Siamese output (or None)
            graph_feat: (B, 300) - frozen graph features
            surface_feat: (B, 256) - frozen surface features
            thermo_feat: (B, 25) - tabular features

        Returns:
            predictions: (B, n_properties)
            aux: dict with routing weights
        """
        modalities = []

        if self.use_images and vit_feat is not None:
            modalities.append(self.vit_proj(vit_feat))
        if self.use_images and siamese_feat is not None:
            modalities.append(self.siamese_proj(siamese_feat))

        modalities.append(self.graph_proj(graph_feat))
        modalities.append(self.surface_proj(surface_feat))
        modalities.append(self.tabular_proj(thermo_feat))

        # Pad to expected number of modalities if some are missing
        while len(modalities) < self.n_modalities:
            modalities.insert(0, torch.zeros_like(modalities[-1]))

        # Fusion
        per_prop_fused, routing_weights, cross_attn_aux = self.fusion(modalities)

        # Prediction
        predictions = []
        for p in range(self.n_properties):
            shared = self.shared_head(per_prop_fused[:, p])
            pred = self.property_heads[p](shared)
            predictions.append(pred)

        predictions = torch.cat(predictions, dim=-1)

        aux = {
            "routing_weights": routing_weights,
            "cross_attn": cross_attn_aux,
        }
        return predictions, aux


def distillation_loss(student_preds, teacher_preds, targets, mask,
                      alpha=0.5, temperature=1.0):
    """Knowledge distillation loss combining ground truth and teacher predictions.

    Solution 6: Uses the v4 model as a teacher to regularize v5-Lite.

    Args:
        student_preds: (B, 7) v5-Lite predictions
        teacher_preds: (B, 7) v4 teacher predictions (detached)
        targets: (B, 7) ground truth
        mask: (B, 7) boolean mask for available labels
        alpha: weight for ground truth loss (1-alpha for teacher loss)
        temperature: softening temperature (1.0 = no softening)

    Returns:
        total_loss: weighted combination of GT and teacher losses
        aux: dict with individual loss components
    """
    mask_float = mask.float()
    n_valid = mask_float.sum()

    if n_valid == 0:
        return torch.tensor(0.0, device=student_preds.device, requires_grad=True), {}

    # Ground truth loss (only for labeled samples)
    gt_loss = ((student_preds - targets) ** 2 * mask_float).sum() / n_valid

    # Teacher loss (for ALL samples, teacher always has predictions)
    # The teacher provides soft targets even for unlabeled properties
    teacher_loss = F.mse_loss(student_preds, teacher_preds.detach())

    # Combined
    total = alpha * gt_loss + (1 - alpha) * teacher_loss

    aux = {
        "gt_loss": gt_loss.item(),
        "teacher_loss": teacher_loss.item(),
        "total_loss": total.item(),
        "alpha": alpha,
    }
    return total, aux

"""COSMOBridge v4 + Image Path: Minimal extension of the proven v4 architecture.

Takes the EXACT v4 model (R²=0.810) and adds images as a 3rd frozen path.
The router learns to blend 3 paths instead of 2:

    Path A: GBH fusion (frozen, pre-computed) → preds_fusion (7D)
    Path B: Chemprop direct (frozen, pre-computed) → preds_chemprop (7D)
    Path C: Image pathway (frozen ViT + tiny head) → preds_image (7D)   ← NEW

    Router: [graph_fp, surface_fp, thermo_feat, image_feat] → α_A, α_B, α_C (softmax)
    Final: α_A * preds_fusion + α_B * preds_chemprop + α_C * preds_image

Key design: The image path is a SEPARATE frozen encoder + tiny trainable head,
exactly matching how v4 treats Paths A and B. Only the router + image head
are trainable (~50K params total). No architectural changes to the proven v4 core.

This is the cleanest way to answer: "do images help on top of v4?"
"""

import torch
import torch.nn as nn
import numpy as np


V3_INIT_LOGITS = torch.tensor([0.36, 0.39, 0.36, 0.42, 0.45, 0.37, 0.69])
V3_INIT_LOGITS = torch.log(V3_INIT_LOGITS / (1 - V3_INIT_LOGITS))


class ImageHead(nn.Module):
    """Tiny prediction head for the image pathway.

    Takes frozen ViT features (192D) and produces 7 property predictions.
    This is pre-trained in isolation, then frozen, so the router
    sees calibrated predictions from all 3 paths.
    """

    def __init__(self, image_dim=192, n_properties=7, dropout=0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(image_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_properties),
        )

    def forward(self, image_feat):
        return self.head(image_feat)


class ThreePathRouter(nn.Module):
    """3-way per-molecule router: extends v4's 2-way router to 3 paths.

    Input: [graph_fp(300), surface_fp(256), thermo_feat(25), image_feat(192)] = 773D
    Output: 3 softmax weights per property → (B, 7, 3)
    """

    def __init__(self, graph_dim=300, surface_dim=256, thermo_dim=25,
                 image_dim=192, hidden=64, n_properties=7, dropout=0.3):
        super().__init__()
        input_dim = graph_dim + surface_dim + thermo_dim + image_dim

        self.router = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, n_properties * 3),  # 3 paths per property
        )

        # Initialize: start with v4-like weights (Path A and B dominant, Path C low)
        with torch.no_grad():
            self.router[-1].weight.zero_()
            # Initialize bias: [fusion_logit, chemprop_logit, image_logit] per property
            bias = torch.zeros(n_properties * 3)
            for p in range(n_properties):
                v3_alpha = torch.sigmoid(V3_INIT_LOGITS[p]).item()
                # Convert 2-way alpha to 3-way logits (image starts near 0)
                bias[p * 3 + 0] = np.log(v3_alpha + 1e-8)          # fusion
                bias[p * 3 + 1] = np.log(1 - v3_alpha + 1e-8)      # chemprop
                bias[p * 3 + 2] = -2.0                               # image (low initially)
            self.router[-1].bias.copy_(bias)

        self.n_properties = n_properties

    def forward(self, graph_fp, surface_fp, thermo_feat, image_feat):
        x = torch.cat([graph_fp, surface_fp, thermo_feat, image_feat], dim=-1)
        logits = self.router(x)  # (B, 21) = (B, 7*3)
        logits = logits.view(-1, self.n_properties, 3)  # (B, 7, 3)
        weights = torch.softmax(logits, dim=-1)  # (B, 7, 3)
        return weights, logits


class COSMOBridgeV4PlusImage(nn.Module):
    """V4 + Image: the proven v4 model with images as a 3rd pathway.

    All 3 path predictions are pre-computed and frozen.
    Only the 3-way router is trainable (~50K params).

    Parameters
    ----------
    graph_dim, surface_dim, thermo_dim, image_dim : int
        Feature dimensions for the router input.
    hidden : int
        Router hidden dimension.
    n_properties : int
        Number of target properties.
    dropout : float
        Dropout rate.
    """

    def __init__(self, graph_dim=300, surface_dim=256, thermo_dim=25,
                 image_dim=192, hidden=64, n_properties=7, dropout=0.3):
        super().__init__()
        self.router = ThreePathRouter(
            graph_dim=graph_dim, surface_dim=surface_dim,
            thermo_dim=thermo_dim, image_dim=image_dim,
            hidden=hidden, n_properties=n_properties, dropout=dropout,
        )

    def forward(self, graph_fp, surface_fp, thermo_feat, image_feat,
                preds_fusion, preds_chemprop, preds_image):
        """
        Args:
            graph_fp: (B, 300)
            surface_fp: (B, 256)
            thermo_feat: (B, 25)
            image_feat: (B, 192) - frozen ViT features
            preds_fusion: (B, 7) - frozen Path A predictions
            preds_chemprop: (B, 7) - frozen Path B predictions
            preds_image: (B, 7) - frozen Path C predictions

        Returns:
            predictions: (B, 7) - 3-way blended predictions
            aux: dict with routing weights
        """
        weights, logits = self.router(graph_fp, surface_fp, thermo_feat, image_feat)
        # weights: (B, 7, 3)

        # Stack path predictions: (B, 7, 3)
        paths = torch.stack([preds_fusion, preds_chemprop, preds_image], dim=-1)

        # Weighted sum
        predictions = (paths * weights).sum(dim=-1)  # (B, 7)

        return predictions, {
            "weights": weights.detach(),   # (B, 7, 3)
            "logits": logits.detach(),     # (B, 7, 3)
            "alpha_fusion": weights[:, :, 0].detach(),
            "alpha_chemprop": weights[:, :, 1].detach(),
            "alpha_image": weights[:, :, 2].detach(),
        }

    def anchor_loss(self, logits):
        """Regularize router to stay near v4 initialization."""
        init_bias = self.router.router[-1].bias.detach()
        B = logits.shape[0]
        init_expanded = init_bias.unsqueeze(0).expand(B, -1)
        return ((logits.view(B, -1) - init_expanded) ** 2).mean()

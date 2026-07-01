"""Multi-Task Supervised ViT: Joint rotation prediction + property prediction.

Combines V-JEPA's rotation objective with direct property prediction,
producing features that encode BOTH molecular geometry AND chemistry.

Architecture:
    Input: 36 COSMO rotation frames per molecule
    Encoder: ViT-Tiny (shared, 192D)

    Head 1 (V-JEPA): Predict masked view representations → geometry learning
    Head 2 (Property): Predict 7 thermodynamic properties → chemistry learning
    Head 3 (Contrastive): Same molecule = similar, different = dissimilar

    Loss = α * L_vjepa + β * L_property + γ * L_contrastive

The key insight: V-JEPA alone learns shape but not chemistry.
Property prediction alone overfits on 19 ILs.
Joint training forces features to encode BOTH, with V-JEPA as regularizer.
"""

import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedViT(nn.Module):
    """Multi-task ViT: V-JEPA + property prediction + contrastive.

    Parameters
    ----------
    encoder : nn.Module
        ViT encoder mapping (B, C, H, W) → (B, embed_dim).
    embed_dim : int
        Encoder output dimension.
    n_properties : int
        Number of property targets.
    n_views : int
        Number of rotation views per molecule.
    predictor_dim : int
        V-JEPA predictor hidden dimension.
    mask_ratio : tuple
        (min, max) fraction of views to mask for V-JEPA.
    ema_decay : float
        EMA decay for V-JEPA target encoder.
    alpha : float
        Weight for V-JEPA loss.
    beta : float
        Weight for property prediction loss.
    gamma : float
        Weight for contrastive loss.
    """

    def __init__(self, encoder, embed_dim=192, n_properties=7, n_views=36,
                 predictor_dim=96, mask_ratio=(0.6, 0.8), ema_decay=0.996,
                 alpha=1.0, beta=0.5, gamma=0.1):
        super().__init__()
        self.n_views = n_views
        self.mask_ratio = mask_ratio
        self.ema_decay = ema_decay
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # Shared encoder
        self.encoder = encoder

        # ── Head 1: V-JEPA predictor ──
        self.target_encoder = copy.deepcopy(encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.predictor = nn.Sequential(
            nn.Linear(embed_dim, predictor_dim),
            nn.GELU(),
            nn.Linear(predictor_dim, embed_dim),
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # ── Head 2: Property prediction ──
        self.property_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_properties),
        )

        # ── Head 3: Contrastive projection ──
        self.contrastive_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 64),
        )
        self.temperature = 0.1

    @torch.no_grad()
    def update_target(self):
        for pt, pc in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            pt.data.mul_(self.ema_decay).add_(pc.data, alpha=1-self.ema_decay)

    def encode_views(self, views, device):
        """Encode all views with shared encoder. (B, V, C, H, W) → (B, V, D)"""
        B, V = views.shape[:2]
        flat = views.reshape(B*V, *views.shape[2:])
        tokens = self.encoder(flat)
        return tokens.reshape(B, V, -1)

    def vjepa_loss(self, views, device):
        """V-JEPA masked prediction loss."""
        B, V = views.shape[:2]

        # Random mask
        ratio = random.uniform(*self.mask_ratio)
        n_mask = max(1, int(V * ratio))
        n_vis = V - n_mask

        ctx_idx = []
        mask_idx = []
        for _ in range(B):
            perm = torch.randperm(V, device=device)
            ctx_idx.append(perm[:n_vis])
            mask_idx.append(perm[n_vis:])
        ctx_idx = torch.stack(ctx_idx)
        mask_idx = torch.stack(mask_idx)

        # Target: encode ALL views with EMA encoder
        with torch.no_grad():
            flat = views.reshape(B*V, *views.shape[2:])
            all_tokens = self.target_encoder(flat).reshape(B, V, -1)
            target = torch.stack([all_tokens[b, mask_idx[b]] for b in range(B)])

        # Context: encode visible views
        vis_views = torch.stack([views[b, ctx_idx[b]] for b in range(B)])
        flat_vis = vis_views.reshape(B*n_vis, *views.shape[2:])
        ctx_tokens = self.encoder(flat_vis).reshape(B, n_vis, -1)

        # Predict masked from context (simplified predictor)
        ctx_mean = ctx_tokens.mean(dim=1)  # (B, D)
        predicted = self.predictor(ctx_mean).unsqueeze(1).expand(-1, n_mask, -1)

        # Loss
        pred_norm = F.layer_norm(predicted, [predicted.shape[-1]])
        tgt_norm = F.layer_norm(target, [target.shape[-1]])
        return F.smooth_l1_loss(pred_norm, tgt_norm)

    def property_loss(self, views, targets, masks):
        """Property prediction from mean view encoding."""
        B, V = views.shape[:2]
        flat = views.reshape(B*V, *views.shape[2:])
        all_tokens = self.encoder(flat).reshape(B, V, -1)
        mol_embed = all_tokens.mean(dim=1)  # (B, D)

        preds = self.property_head(mol_embed)

        # Masked MSE
        m = masks.float()
        n = m.sum()
        if n == 0:
            return torch.tensor(0.0, device=views.device, requires_grad=True), preds
        loss = ((preds - targets)**2 * m).sum() / n
        return loss, preds

    def contrastive_loss(self, views):
        """Contrastive: different views of same molecule should be similar."""
        B, V = views.shape[:2]
        if B < 2:
            return torch.tensor(0.0, device=views.device, requires_grad=True)

        # Take 2 random views per molecule
        idx1 = torch.randint(V, (B,))
        idx2 = torch.randint(V, (B,))
        v1 = torch.stack([views[b, idx1[b]] for b in range(B)])
        v2 = torch.stack([views[b, idx2[b]] for b in range(B)])

        z1 = F.normalize(self.contrastive_proj(self.encoder(v1)), dim=1)
        z2 = F.normalize(self.contrastive_proj(self.encoder(v2)), dim=1)

        sim = torch.mm(z1, z2.t()) / self.temperature
        labels = torch.arange(B, device=views.device)
        return F.cross_entropy(sim, labels)

    def forward(self, views, targets=None, masks=None):
        """
        Args:
            views: (B, V, C, H, W) rotation frames
            targets: (B, 7) property targets (optional)
            masks: (B, 7) which properties have labels (optional)

        Returns:
            total_loss, aux dict
        """
        device = views.device
        losses = {}

        # V-JEPA loss
        l_vjepa = self.vjepa_loss(views, device)
        losses["vjepa"] = l_vjepa.item()

        # Property loss
        if targets is not None and masks is not None:
            l_prop, preds = self.property_loss(views, targets, masks)
            losses["property"] = l_prop.item()
        else:
            l_prop = torch.tensor(0.0, device=device)
            preds = None

        # Contrastive loss
        l_contra = self.contrastive_loss(views)
        losses["contrastive"] = l_contra.item()

        total = self.alpha * l_vjepa + self.beta * l_prop + self.gamma * l_contra
        losses["total"] = total.item()

        return total, {"losses": losses, "preds": preds}

    def extract_features(self, views):
        """Extract molecule-level features for downstream use."""
        B, V = views.shape[:2]
        flat = views.reshape(B*V, *views.shape[2:])
        all_tokens = self.encoder(flat).reshape(B, V, -1)
        return all_tokens.mean(dim=1)  # (B, D)

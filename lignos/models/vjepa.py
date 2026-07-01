"""V-JEPA (Video Joint Embedding Predictive Architecture) for COSMO Surfaces.

Adapts Meta's V-JEPA framework to molecular COSMO surface "videos" (36 rotation
frames). Instead of predicting masked pixel patches, V-JEPA predicts masked
view representations in latent space -- learning a "world model" of how the
molecular surface looks from any angle.

Key advantages over SimCLR for our use case:
    1. No negative pairs needed (SimCLR has only 301 molecules = tiny batch)
    2. Predicts in latent space (more semantic than pixel reconstruction)
    3. Masking rotation views forces learning of 3D surface geometry
    4. The predictor learns rotational transformations as physical priors

Architecture:
    Context encoder:  ViT-Tiny processes VISIBLE rotation views -> latent tokens
    Target encoder:   EMA copy of context encoder processes ALL views -> target tokens
    Predictor:        Small transformer predicts MASKED view tokens from context + mask tokens

Training:
    1. Randomly mask 60-80% of the 36 rotation views
    2. Context encoder sees only unmasked views
    3. Target encoder (EMA, no gradient) sees all views
    4. Predictor predicts masked view tokens from context
    5. Loss = MSE(predicted_tokens, target_tokens) in latent space

References:
    Bardes et al., "V-JEPA: Latent Video Prediction for Visual Representation Learning", 2024
"""

import copy
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class VJEPAPredictor(nn.Module):
    """Lightweight transformer predictor for V-JEPA.

    Takes context tokens (visible views) + mask tokens (learnable),
    and predicts the latent representations of masked views.

    Parameters
    ----------
    embed_dim : int
        Dimension matching the encoder output.
    predictor_dim : int
        Internal dimension of the predictor (can be smaller than embed_dim).
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of transformer layers.
    n_views : int
        Total number of rotation views.
    dropout : float
        Dropout probability.
    """

    def __init__(self, embed_dim=192, predictor_dim=96, n_heads=4,
                 n_layers=2, n_views=36, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.predictor_dim = predictor_dim
        self.n_views = n_views

        # Project from encoder dim to predictor dim
        self.input_proj = nn.Linear(embed_dim, predictor_dim)

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))

        # View positional embeddings (encode rotation angle)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_views, predictor_dim))

        # Predictor transformer
        layer = nn.TransformerEncoderLayer(
            d_model=predictor_dim, nhead=n_heads,
            dim_feedforward=predictor_dim * 2,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(predictor_dim)

        # Project back to encoder dim for loss computation
        self.output_proj = nn.Linear(predictor_dim, embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, context_tokens, context_indices, mask_indices):
        """
        Args:
            context_tokens: (B, n_visible, embed_dim) - encoder output for visible views
            context_indices: (B, n_visible) - which view indices are visible
            mask_indices: (B, n_masked) - which view indices are masked

        Returns:
            predicted: (B, n_masked, embed_dim) - predicted latent tokens for masked views
        """
        B = context_tokens.shape[0]
        n_visible = context_tokens.shape[1]
        n_masked = mask_indices.shape[1]

        # Project context tokens
        ctx = self.input_proj(context_tokens)  # (B, n_visible, pred_dim)

        # Create mask tokens (clone to avoid in-place on leaf)
        mask_tok = self.mask_token.expand(B, n_masked, -1).clone()  # (B, n_masked, pred_dim)

        # Add positional embeddings based on view index (use gather for batched indexing)
        ctx_pos = torch.stack([self.pos_embed[0, context_indices[b]] for b in range(B)])
        mask_pos = torch.stack([self.pos_embed[0, mask_indices[b]] for b in range(B)])
        ctx = ctx + ctx_pos
        mask_tok = mask_tok + mask_pos

        # Concatenate: [context_tokens, mask_tokens]
        tokens = torch.cat([ctx, mask_tok], dim=1)  # (B, n_visible + n_masked, pred_dim)

        # Predict
        predicted = self.transformer(tokens)
        predicted = self.norm(predicted)

        # Extract only the masked positions
        predicted_masked = predicted[:, n_visible:]  # (B, n_masked, pred_dim)

        # Project back to encoder dim
        return self.output_proj(predicted_masked)  # (B, n_masked, embed_dim)


class COSMO_VJEPA(nn.Module):
    """V-JEPA for COSMO surface rotation videos.

    Parameters
    ----------
    encoder : nn.Module
        ViT-Tiny encoder that maps (B, 3, H, W) -> (B, embed_dim).
    embed_dim : int
        Encoder output dimension.
    predictor_dim : int
        Predictor internal dimension.
    n_views : int
        Total number of rotation views per molecule.
    mask_ratio : tuple
        (min, max) fraction of views to mask. Default (0.6, 0.8).
    ema_decay : float
        Exponential moving average decay for target encoder. Default 0.996.
    predictor_layers : int
        Number of transformer layers in predictor.
    """

    def __init__(self, encoder, embed_dim=192, predictor_dim=96,
                 n_views=36, mask_ratio=(0.6, 0.8), ema_decay=0.996,
                 predictor_layers=2):
        super().__init__()
        self.n_views = n_views
        self.mask_ratio = mask_ratio
        self.ema_decay = ema_decay

        # Context encoder (trained with gradients)
        self.context_encoder = encoder

        # Target encoder (EMA copy, no gradients)
        self.target_encoder = copy.deepcopy(encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        # Predictor
        self.predictor = VJEPAPredictor(
            embed_dim=embed_dim,
            predictor_dim=predictor_dim,
            n_heads=4,
            n_layers=predictor_layers,
            n_views=n_views,
        )

    @torch.no_grad()
    def update_target_encoder(self):
        """Update target encoder with exponential moving average."""
        for param_t, param_c in zip(self.target_encoder.parameters(),
                                     self.context_encoder.parameters()):
            param_t.data.mul_(self.ema_decay).add_(param_c.data, alpha=1 - self.ema_decay)

    def generate_mask(self, batch_size, device):
        """Generate random view masks for a batch.

        Returns:
            context_indices: (B, n_visible) - indices of visible views
            mask_indices: (B, n_masked) - indices of masked views
        """
        # Random mask ratio within range
        ratio = random.uniform(*self.mask_ratio)
        n_masked = max(1, int(self.n_views * ratio))
        n_visible = self.n_views - n_masked

        context_indices = []
        mask_indices = []

        for _ in range(batch_size):
            perm = torch.randperm(self.n_views, device=device)
            context_indices.append(perm[:n_visible].sort().values)
            mask_indices.append(perm[n_visible:].sort().values)

        return (torch.stack(context_indices),   # (B, n_visible)
                torch.stack(mask_indices))       # (B, n_masked)

    def forward(self, views):
        """
        Args:
            views: (B, n_views, 3, H, W) - all rotation frames

        Returns:
            loss: scalar JEPA prediction loss
            aux: dict with diagnostics
        """
        B, V = views.shape[:2]
        device = views.device

        # Generate mask
        ctx_idx, mask_idx = self.generate_mask(B, device)
        n_visible = ctx_idx.shape[1]
        n_masked = mask_idx.shape[1]

        # Encode ALL views with target encoder (no gradient)
        with torch.no_grad():
            flat = views.reshape(B * V, *views.shape[2:])
            all_tokens = self.target_encoder(flat)  # (B*V, embed_dim)
            all_tokens = all_tokens.reshape(B, V, -1)  # (B, V, embed_dim)

            # Extract target tokens for masked views
            target_tokens = torch.stack([
                all_tokens[b, mask_idx[b]] for b in range(B)
            ])  # (B, n_masked, embed_dim)

        # Encode only VISIBLE views with context encoder (with gradient)
        visible_views = torch.stack([
            views[b, ctx_idx[b]] for b in range(B)
        ])  # (B, n_visible, 3, H, W)

        flat_visible = visible_views.reshape(B * n_visible, *views.shape[2:])
        context_tokens = self.context_encoder(flat_visible)  # (B*n_visible, embed_dim)
        context_tokens = context_tokens.reshape(B, n_visible, -1)  # (B, n_visible, embed_dim)

        # Predict masked view tokens
        predicted = self.predictor(context_tokens, ctx_idx, mask_idx)  # (B, n_masked, embed_dim)

        # Loss: MSE in latent space (smooth L1 is also common)
        # Normalize both predicted and target for stability
        predicted_norm = F.layer_norm(predicted, [predicted.shape[-1]])
        target_norm = F.layer_norm(target_tokens, [target_tokens.shape[-1]])

        loss = F.smooth_l1_loss(predicted_norm, target_norm)

        # Diagnostics
        with torch.no_grad():
            cosine_sim = F.cosine_similarity(
                predicted.reshape(-1, predicted.shape[-1]),
                target_tokens.reshape(-1, target_tokens.shape[-1]),
                dim=-1
            ).mean()

        aux = {
            "loss": loss.item(),
            "cosine_sim": cosine_sim.item(),
            "n_visible": n_visible,
            "n_masked": n_masked,
            "mask_ratio": n_masked / V,
        }

        return loss, aux


class COSMOViewDataset(torch.utils.data.Dataset):
    """Dataset that loads all rotation views for each molecule.

    For V-JEPA, we need ALL views per molecule (not just pairs like SimCLR).

    Parameters
    ----------
    mol_ids : list[str]
        Molecule identifiers.
    mol_to_dir : dict
        Mapping from mol_id to Path of frame directory.
    transform : callable
        Image transform.
    n_views : int
        Number of views to load per molecule.
    """

    def __init__(self, mol_ids, mol_to_dir, transform=None, n_views=36):
        self.mol_ids = mol_ids
        self.mol_to_dir = mol_to_dir
        self.transform = transform
        self.n_views = n_views

    def __len__(self):
        return len(self.mol_ids)

    def __getitem__(self, idx):
        from PIL import Image

        mol_id = self.mol_ids[idx]
        frame_dir = self.mol_to_dir[mol_id]
        frames = sorted(frame_dir.glob("frame_*.png"))

        views = []
        for i in range(min(self.n_views, len(frames))):
            img = Image.open(frames[i]).convert("RGB")
            if self.transform:
                img = self.transform(img)
            views.append(img)

        # Pad if needed
        while len(views) < self.n_views:
            views.append(views[-1].clone())

        return torch.stack(views), mol_id  # (n_views, 3, H, W)

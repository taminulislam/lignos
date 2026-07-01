"""Multi-View Vision Transformer for COSMO Surface Images.

Processes 36 rotation frames of a molecular COSMO surface through a shared
ViT-Tiny encoder, then aggregates with view-level self-attention and
learnable query pooling.

Architecture:
    36 COSMO frames (224x224x3)
        |
        v
    ViT-Tiny (shared weights, 6 layers, 192D) --> 36 x [CLS] tokens
        |
        v
    View-Level Self-Attention (2 layers, 4 heads)
        |
        v
    Learnable Query Pooling --> single 192D molecule embedding
"""

import torch
import torch.nn as nn
import math


class PatchEmbedding(nn.Module):
    """Convert image into sequence of patch embeddings."""

    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=192):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W) -> (B, n_patches, embed_dim)
        return self.proj(x).flatten(2).transpose(1, 2)


class MultiViewViT(nn.Module):
    """Multi-View ViT-Tiny with view-level self-attention.

    Parameters
    ----------
    n_views : int
        Number of rotation views per molecule (default 36).
    embed_dim : int
        ViT embedding dimension (192 for ViT-Tiny).
    img_size : int
        Input image size (default 224).
    patch_size : int
        Patch size for ViT (default 16).
    in_channels : int
        Number of input channels (3 for RGB, 6 for COSMO+EP dual-channel).
    n_vit_layers : int
        Number of transformer layers in the ViT encoder.
    n_vit_heads : int
        Number of attention heads in ViT.
    n_view_layers : int
        Number of self-attention layers for view aggregation.
    n_view_heads : int
        Number of attention heads for view aggregation.
    mlp_ratio : float
        MLP expansion ratio in ViT.
    dropout : float
        Dropout probability.
    stochastic_depth : float
        Stochastic depth drop rate.
    """

    def __init__(
        self,
        n_views=36,
        embed_dim=192,
        img_size=224,
        patch_size=16,
        in_channels=3,
        n_vit_layers=6,
        n_vit_heads=3,
        n_view_layers=2,
        n_view_heads=4,
        mlp_ratio=4,
        dropout=0.1,
        stochastic_depth=0.1,
    ):
        super().__init__()
        self.n_views = n_views
        self.embed_dim = embed_dim

        # ── Patch Embedding ──
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches

        # ── Positional Embeddings + CLS Token ──
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)

        # ── ViT Encoder ──
        dpr = [x.item() for x in torch.linspace(0, stochastic_depth, n_vit_layers)]
        self.vit_blocks = nn.ModuleList([
            ViTBlock(embed_dim, n_vit_heads, mlp_ratio, dropout, drop_path=dpr[i])
            for i in range(n_vit_layers)
        ])
        self.vit_norm = nn.LayerNorm(embed_dim)

        # ── View-Level Self-Attention ──
        self.view_pos_embed = nn.Parameter(torch.zeros(1, n_views, embed_dim))
        view_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_view_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.view_attention = nn.TransformerEncoder(view_layer, num_layers=n_view_layers)
        self.view_norm = nn.LayerNorm(embed_dim)

        # ── Learnable Query Pooling ──
        self.pool_query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pool_attn = nn.MultiheadAttention(
            embed_dim, n_view_heads, dropout=dropout, batch_first=True
        )
        self.pool_norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.view_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.pool_query, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def encode_single_view(self, x):
        """Encode one image through ViT. (B, C, H, W) -> (B, embed_dim)."""
        patches = self.patch_embed(x)  # (B, N, D)
        B = patches.shape[0]

        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, patches], dim=1)  # (B, N+1, D)
        tokens = self.pos_dropout(tokens + self.pos_embed)

        for block in self.vit_blocks:
            tokens = block(tokens)

        tokens = self.vit_norm(tokens)
        return tokens[:, 0]  # [CLS] token: (B, D)

    def forward(self, views):
        """
        Args:
            views: (B, n_views, C, H, W) - rotation frames per molecule

        Returns:
            embedding: (B, embed_dim) - view-aggregated molecule embedding
            view_weights: (B, n_views) - attention weights per view
        """
        B, V = views.shape[:2]

        # Encode all views with shared weights
        flat = views.reshape(B * V, *views.shape[2:])
        cls_tokens = self.encode_single_view(flat)  # (B*V, D)
        view_tokens = cls_tokens.reshape(B, V, -1)  # (B, V, D)

        # View-level self-attention
        view_tokens = view_tokens + self.view_pos_embed[:, :V]
        attended = self.view_attention(view_tokens)
        attended = self.view_norm(attended)

        # Learnable pooling
        query = self.pool_query.expand(B, -1, -1)
        pooled, attn_weights = self.pool_attn(query, attended, attended)
        pooled = self.pool_norm(pooled)

        embedding = pooled.squeeze(1)           # (B, D)
        view_weights = attn_weights.squeeze(1)  # (B, V)

        return embedding, view_weights

    def encode_views_chunked(self, views, chunk_size=6):
        """Memory-efficient forward: process views in chunks.

        Use this when GPU memory is limited (36 views x 224x224 is large).
        """
        B, V = views.shape[:2]
        all_cls = []

        for i in range(0, V, chunk_size):
            chunk = views[:, i : i + chunk_size]
            flat = chunk.reshape(-1, *chunk.shape[2:])
            cls = self.encode_single_view(flat)
            all_cls.append(cls.reshape(B, -1, self.embed_dim))

        view_tokens = torch.cat(all_cls, dim=1)  # (B, V, D)

        # View-level self-attention + pooling (same as forward)
        view_tokens = view_tokens + self.view_pos_embed[:, :V]
        attended = self.view_attention(view_tokens)
        attended = self.view_norm(attended)

        query = self.pool_query.expand(B, -1, -1)
        pooled, attn_weights = self.pool_attn(query, attended, attended)
        pooled = self.pool_norm(pooled)

        return pooled.squeeze(1), attn_weights.squeeze(1)


class ViTBlock(nn.Module):
    """Transformer block with pre-norm and optional stochastic depth."""

    def __init__(self, dim, n_heads, mlp_ratio=4, dropout=0.1, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + self.drop_path(attn_out)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DropPath(nn.Module):
    """Stochastic depth (drop path) regularization."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.rand(shape, device=x.device) < keep_prob
        return x * mask / keep_prob

"""Cation-Anion Siamese Network with Cross-Attention.

Processes separate COSMO surface images of the cation and anion through a
shared encoder, then models their interaction via bidirectional cross-attention.

Architecture:
    Cation COSMO (224x224)       Anion COSMO (224x224)
           |                            |
           v                            v
      SharedCNN (4 layers)         SharedCNN (shared weights)
           |                            |
           v                            v
      (B, 49, D) tokens           (B, 49, D) tokens
           |                            |
           +---- Cross-Attention -------+
           |                            |
           v                            v
      cation_attended              anion_attended
           |                            |
           +-------- Interact ----------+
                       |
                       v
              192D interaction embedding

The Hadamard product (cat * an) captures charge complementarity:
positive regions on cation meeting negative regions on anion.
"""

import torch
import torch.nn as nn


class SharedCNNEncoder(nn.Module):
    """Lightweight CNN for encoding single-ion COSMO images into spatial tokens."""

    def __init__(self, in_channels=3, channels=(32, 64, 128, 256), dropout=0.1):
        super().__init__()
        layers = []
        c_in = in_channels
        for c_out in channels:
            layers.extend([
                nn.Conv2d(c_in, c_out, 3, stride=2, padding=1),
                nn.BatchNorm2d(c_out),
                nn.GELU(),
                nn.Dropout2d(dropout),
            ])
            c_in = c_out
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(7)  # -> (B, C, 7, 7) = 49 spatial tokens
        self.out_channels = channels[-1]

    def forward(self, x):
        """(B, C, H, W) -> (B, 49, out_channels)."""
        feat = self.pool(self.features(x))  # (B, C, 7, 7)
        B, C, H, W = feat.shape
        return feat.reshape(B, C, H * W).permute(0, 2, 1)  # (B, 49, C)


class CationAnionSiamese(nn.Module):
    """Siamese architecture for modeling cation-anion interactions.

    Parameters
    ----------
    embed_dim : int
        Output embedding dimension.
    n_heads : int
        Number of cross-attention heads.
    encoder_channels : list[int]
        Channel sizes for the shared CNN encoder.
    in_channels : int
        Input image channels (3 for RGB).
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        embed_dim=192,
        n_heads=4,
        encoder_channels=(32, 64, 128, 256),
        in_channels=3,
        dropout=0.2,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Shared encoder for cation and anion
        self.encoder = SharedCNNEncoder(in_channels, encoder_channels, dropout)
        enc_dim = self.encoder.out_channels

        # Project to common dimension
        self.spatial_proj = nn.Linear(enc_dim, embed_dim)

        # Bidirectional cross-attention
        self.cat_to_an = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.an_to_cat = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=dropout, batch_first=True
        )

        self.ln_cat = nn.LayerNorm(embed_dim)
        self.ln_an = nn.LayerNorm(embed_dim)

        # Feed-forward after cross-attention
        self.ffn_cat = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.ffn_an = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.ln_cat2 = nn.LayerNorm(embed_dim)
        self.ln_an2 = nn.LayerNorm(embed_dim)

        # Interaction MLP: concat + hadamard + difference = 4 * embed_dim
        self.interaction = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def encode_ion(self, img):
        """Encode a single ion image into spatial tokens.

        Args:
            img: (B, C, H, W)

        Returns:
            tokens: (B, 49, embed_dim)
        """
        tokens = self.encoder(img)          # (B, 49, enc_dim)
        return self.spatial_proj(tokens)     # (B, 49, embed_dim)

    def forward(self, cation_img, anion_img):
        """
        Args:
            cation_img: (B, C, H, W) - cation COSMO surface image
            anion_img: (B, C, H, W) - anion COSMO surface image

        Returns:
            il_embedding: (B, embed_dim) - ionic liquid interaction embedding
            aux: dict with cross-attention weights for interpretability
        """
        # Encode both ions
        cat_tokens = self.encode_ion(cation_img)  # (B, 49, D)
        an_tokens = self.encode_ion(anion_img)     # (B, 49, D)

        # Cross-attention: each ion attends to the other's surface
        cat_attn, cat_weights = self.cat_to_an(cat_tokens, an_tokens, an_tokens)
        an_attn, an_weights = self.an_to_cat(an_tokens, cat_tokens, cat_tokens)

        # Residual + norm + FFN
        cat_out = self.ln_cat(cat_tokens + cat_attn)
        an_out = self.ln_an(an_tokens + an_attn)

        cat_out = self.ln_cat2(cat_out + self.ffn_cat(cat_out))
        an_out = self.ln_an2(an_out + self.ffn_an(an_out))

        # Pool spatial tokens
        cat_pooled = cat_out.mean(dim=1)  # (B, D)
        an_pooled = an_out.mean(dim=1)    # (B, D)

        # Interaction features
        interaction_input = torch.cat([
            cat_pooled,
            an_pooled,
            cat_pooled * an_pooled,   # charge complementarity
            cat_pooled - an_pooled,   # asymmetry
        ], dim=-1)  # (B, 4*D)

        il_embedding = self.interaction(interaction_input)  # (B, D)

        aux = {
            "cat_to_an_weights": cat_weights.detach(),  # (B, n_heads, 49, 49)
            "an_to_cat_weights": an_weights.detach(),
        }

        return il_embedding, aux

"""COSMOBridge v5: Full Multimodal Architecture.

Fuses 5 modalities with cross-modal attention and per-property routing:

    1. Multi-View ViT-Tiny (SimCLR pre-trained) -- COSMO surface images
    2. Cation-Anion Siamese -- ion interaction from separate images
    3. Chemprop D-MPNN (frozen) -- molecular graph features
    4. PointNet (frozen) -- 3D COSMO point cloud features
    5. Tabular MLP -- thermodynamic + surface descriptors

Training: 3-stage protocol
    Stage 1: Freeze all encoders, train fusion + heads
    Stage 2: Unfreeze image encoders (ViT + Siamese), differential LR
    Stage 3: Full fine-tuning with low LR (optional)
"""

import torch
import torch.nn as nn

from .multiview_vit import MultiViewViT
from .siamese_encoder import CationAnionSiamese
from .cross_modal_attention import CrossModalFusion


class COSMOBridgeV5(nn.Module):
    """Full multimodal architecture for IL thermodynamic property prediction.

    Parameters
    ----------
    embed_dim : int
        Common embedding dimension for fusion (default 256).
    n_properties : int
        Number of target properties (default 7).
    n_views : int
        Number of COSMO rotation views (default 36).
    graph_dim : int
        Dimension of frozen Chemprop graph fingerprint (default 300).
    surface_dim : int
        Dimension of frozen PointNet surface features (default 256).
    thermo_dim : int
        Dimension of thermodynamic + descriptor features (default 25).
    vit_embed_dim : int
        ViT-Tiny embedding dimension (default 192).
    siamese_embed_dim : int
        Siamese encoder embedding dimension (default 192).
    n_cross_attn_heads : int
        Number of heads in cross-modal attention (default 4).
    dropout : float
        Dropout probability (default 0.2).
    """

    def __init__(
        self,
        embed_dim=256,
        n_properties=7,
        n_views=36,
        graph_dim=300,
        surface_dim=256,
        thermo_dim=25,
        vit_embed_dim=192,
        siamese_embed_dim=192,
        n_cross_attn_heads=4,
        dropout=0.2,
        vit_in_channels=3,
        siamese_in_channels=3,
        siamese_channels=(32, 64, 128, 256),
    ):
        super().__init__()
        self.n_properties = n_properties
        self.embed_dim = embed_dim

        # ══════════════════════════════════════════════
        # MODALITY ENCODERS
        # ══════════════════════════════════════════════

        # 1. Multi-View ViT-Tiny (SimCLR pre-trained)
        self.multiview_vit = MultiViewViT(
            n_views=n_views,
            embed_dim=vit_embed_dim,
            in_channels=vit_in_channels,
            dropout=dropout * 0.5,
        )
        self.vit_proj = nn.Sequential(
            nn.Linear(vit_embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # 2. Cation-Anion Siamese
        self.siamese = CationAnionSiamese(
            embed_dim=siamese_embed_dim,
            n_heads=n_cross_attn_heads,
            encoder_channels=siamese_channels,
            in_channels=siamese_in_channels,
            dropout=dropout,
        )
        self.siamese_proj = nn.Sequential(
            nn.Linear(siamese_embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # 3. Graph encoder projection (frozen D-MPNN features as input)
        self.graph_proj = nn.Sequential(
            nn.Linear(graph_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 4. Surface encoder projection (frozen PointNet features as input)
        self.surface_proj = nn.Sequential(
            nn.Linear(surface_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 5. Tabular encoder
        self.tabular_encoder = nn.Sequential(
            nn.Linear(thermo_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # ══════════════════════════════════════════════
        # CROSS-MODAL ATTENTION FUSION
        # ══════════════════════════════════════════════

        self.fusion = CrossModalFusion(
            dim=embed_dim,
            n_heads=n_cross_attn_heads,
            n_modalities=5,
            n_properties=n_properties,
            dropout=dropout * 0.5,
        )

        # ══════════════════════════════════════════════
        # MULTI-TASK PREDICTION HEADS
        # ══════════════════════════════════════════════

        # Shared backbone per property
        self.shared_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Property-specific sub-heads
        self.property_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, 64),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(64, 1),
            )
            for _ in range(n_properties)
        ])

    def load_simclr_weights(self, simclr_checkpoint_path):
        """Load pre-trained SimCLR weights into the ViT encoder.

        Only loads the encoder weights, ignoring the projection head.

        Args:
            simclr_checkpoint_path: path to SimCLR checkpoint file.
        """
        import os
        if not os.path.exists(simclr_checkpoint_path):
            print(f"WARNING: SimCLR checkpoint not found: {simclr_checkpoint_path}")
            return

        checkpoint = torch.load(
            simclr_checkpoint_path, map_location="cpu", weights_only=True
        )

        # Unwrap the state dict. Supported container shapes:
        #   - raw state_dict (V-JEPA saves {"encoder_state_dict": {...}})
        #   - SimCLR wrapper: {"model_state_dict": {"encoder.<key>": ...}}
        #   - plain dict of tensors
        if isinstance(checkpoint, dict):
            state = checkpoint.get(
                "encoder_state_dict",
                checkpoint.get("model_state_dict",
                                checkpoint.get("state_dict", checkpoint)),
            )
        else:
            state = checkpoint

        # Normalise keys: strip `encoder.` prefix if SimCLR-style.
        stripped = {}
        for k, v in state.items():
            if k.startswith("encoder."):
                stripped[k[len("encoder."):]] = v
            else:
                stripped[k] = v

        # Remap single-view ViT keys to the MultiViewViT naming convention:
        #   blocks.N.*  -> vit_blocks.N.*
        #   norm.*      -> vit_norm.*
        # Other keys (patch_embed, cls_token, pos_embed) already match.
        remapped = {}
        for k, v in stripped.items():
            if k.startswith("blocks."):
                remapped["vit_" + k] = v
            elif k.startswith("norm."):
                remapped["vit_norm." + k[len("norm."):]] = v
            else:
                remapped[k] = v

        if remapped:
            missing, unexpected = self.multiview_vit.load_state_dict(
                remapped, strict=False
            )
            # Only warn about unexpected keys (they indicate a real mismatch).
            # Missing keys are expected — the multi-view attention layers are
            # not present in the single-view pretraining checkpoint.
            loaded = len(remapped) - len(unexpected)
            print(
                f"Loaded pretrained ViT: {loaded}/{len(remapped)} params matched, "
                f"{len(unexpected)} unexpected keys"
            )
            if unexpected:
                print(f"  unexpected sample: {unexpected[:3]}")
        else:
            print("WARNING: No encoder keys found in pretrained checkpoint")

    def forward(
        self,
        views,
        cation_img,
        anion_img,
        graph_feat,
        surface_feat,
        thermo_feat,
        use_chunked_views=False,
    ):
        """
        Args:
            views: (B, n_views, C, H, W) - multi-view COSMO rotation frames
            cation_img: (B, C, H, W) - cation COSMO surface image
            anion_img: (B, C, H, W) - anion COSMO surface image
            graph_feat: (B, graph_dim) - frozen Chemprop D-MPNN fingerprint
            surface_feat: (B, surface_dim) - frozen PointNet features
            thermo_feat: (B, thermo_dim) - thermodynamic + surface descriptors
            use_chunked_views: bool - process views in memory-efficient chunks

        Returns:
            predictions: (B, n_properties) - predicted property values
            aux: dict with interpretable attention weights and routing info
        """
        # ── Encode all modalities ──
        if use_chunked_views:
            vit_emb, view_weights = self.multiview_vit.encode_views_chunked(views)
        else:
            vit_emb, view_weights = self.multiview_vit(views)

        siamese_emb, siamese_aux = self.siamese(cation_img, anion_img)

        # Project all to common dimension
        m_vit = self.vit_proj(vit_emb)            # (B, embed_dim)
        m_siam = self.siamese_proj(siamese_emb)   # (B, embed_dim)
        m_graph = self.graph_proj(graph_feat)      # (B, embed_dim)
        m_surface = self.surface_proj(surface_feat)  # (B, embed_dim)
        m_tab = self.tabular_encoder(thermo_feat)    # (B, embed_dim)

        # ── Cross-Modal Attention Fusion ──
        per_prop_fused, routing_weights, cross_attn_aux = self.fusion(
            [m_vit, m_siam, m_graph, m_surface, m_tab]
        )  # (B, n_properties, embed_dim), (n_properties, 5)

        # ── Multi-Task Prediction ──
        predictions = []
        for p in range(self.n_properties):
            shared = self.shared_head(per_prop_fused[:, p])  # (B, embed_dim)
            pred = self.property_heads[p](shared)            # (B, 1)
            predictions.append(pred)

        predictions = torch.cat(predictions, dim=-1)  # (B, n_properties)

        aux = {
            "routing_weights": routing_weights,       # (P, 5) per-property modality mix
            "view_weights": view_weights,             # (B, n_views) ViT view attention
            "cross_attn": cross_attn_aux,             # cross-attention weight maps
            "siamese": siamese_aux,                   # ion cross-attention maps
        }

        return predictions, aux

    def get_parameter_groups(self, image_lr=1e-4, fusion_lr=1e-3, head_lr=1e-3):
        """Get parameter groups with differential learning rates.

        For Stage 2 training: image encoders get lower LR.

        Returns:
            list of dicts for torch.optim.AdamW
        """
        image_params = (
            list(self.multiview_vit.parameters())
            + list(self.siamese.parameters())
        )
        fusion_params = (
            list(self.vit_proj.parameters())
            + list(self.siamese_proj.parameters())
            + list(self.graph_proj.parameters())
            + list(self.surface_proj.parameters())
            + list(self.tabular_encoder.parameters())
            + list(self.fusion.parameters())
        )
        head_params = (
            list(self.shared_head.parameters())
            + list(self.property_heads.parameters())
        )

        return [
            {"params": image_params, "lr": image_lr, "name": "image_encoders"},
            {"params": fusion_params, "lr": fusion_lr, "name": "fusion"},
            {"params": head_params, "lr": head_lr, "name": "heads"},
        ]

    def freeze_encoders(self):
        """Freeze all modality encoders (Stage 1 training)."""
        for param in self.multiview_vit.parameters():
            param.requires_grad = False
        for param in self.siamese.parameters():
            param.requires_grad = False
        print("Frozen: ViT + Siamese encoders")

    def unfreeze_image_encoders(self):
        """Unfreeze image encoders only (Stage 2 training)."""
        for param in self.multiview_vit.parameters():
            param.requires_grad = True
        for param in self.siamese.parameters():
            param.requires_grad = True
        print("Unfrozen: ViT + Siamese encoders")

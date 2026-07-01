"""COSMOBridge v5.1: Better Fusion + Descriptor Pathway + Image-Ready.

Fixes the 3 bottlenecks identified by gap analysis:

    1. BETTER FUSION: Replace GBH bilinear with cross-attention fusion
       (Linear LOO R²=0.90 for G_E but v4 GBH only gets 0.77 → fusion is bottleneck)

    2. DESCRIPTOR PATHWAY: Surface descriptors (20D) as a dedicated 3rd path
       (Descriptors alone hit R²=0.85 for P, 0.54 for H_vap — underused in v4)

    3. TEMPERATURE MODULATION: T-conditioned feature mixing
       (Same PointNet features for all T wastes temperature dependence)

    4. IMAGE-READY: Optional 4th pathway for when images prove useful at scale

Architecture:
    Path A: Graph × Surface CROSS-ATTENTION fusion → 7 preds  (improved from GBH)
    Path B: Chemprop direct FFN → 7 preds                     (same as v4)
    Path C: Surface Descriptors + T-modulation → 7 preds       (NEW)
    Path D: Image features → 7 preds                           (optional, for later)

    Router: Per-molecule, per-property softmax over 3-4 paths
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionFusion(nn.Module):
    """Graph-Surface cross-attention: replaces GBH bilinear fusion.

    Graph features attend to surface features and vice versa,
    then combined with a residual MLP. This captures richer
    interactions than element-wise bilinear product.
    """

    def __init__(self, graph_dim=300, surface_dim=256, fused_dim=256,
                 n_heads=4, dropout=0.3):
        super().__init__()
        self.graph_proj = nn.Linear(graph_dim, fused_dim)
        self.surface_proj = nn.Linear(surface_dim, fused_dim)

        # Bidirectional cross-attention
        self.g2s_attn = nn.MultiheadAttention(
            fused_dim, n_heads, dropout=dropout, batch_first=True)
        self.s2g_attn = nn.MultiheadAttention(
            fused_dim, n_heads, dropout=dropout, batch_first=True)

        self.ln_g = nn.LayerNorm(fused_dim)
        self.ln_s = nn.LayerNorm(fused_dim)

        # Fusion MLP (deeper than GBH)
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim * 3, fused_dim),  # concat + hadamard
            nn.LayerNorm(fused_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, fused_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, graph_feat, surface_feat):
        g = self.graph_proj(graph_feat).unsqueeze(1)  # (B, 1, D)
        s = self.surface_proj(surface_feat).unsqueeze(1)  # (B, 1, D)

        # Cross-attention
        g_attended, _ = self.g2s_attn(g, s, s)
        s_attended, _ = self.s2g_attn(s, g, g)

        g_out = self.ln_g(g + g_attended).squeeze(1)  # (B, D)
        s_out = self.ln_s(s + s_attended).squeeze(1)  # (B, D)

        # Combine: concat + hadamard product (captures complementary info)
        combined = torch.cat([g_out, s_out, g_out * s_out], dim=-1)
        return self.fusion(combined)  # (B, fused_dim)


class DescriptorPathway(nn.Module):
    """Dedicated pathway for surface descriptors + temperature modulation.

    Surface descriptors (20D) contain curvature, ESP statistics, etc.
    Temperature modulation allows the same descriptors to produce
    different predictions at different temperatures.
    """

    def __init__(self, desc_dim=20, thermo_dim=5, hidden=64,
                 n_properties=7, dropout=0.3):
        super().__init__()
        # Temperature embedding
        self.temp_embed = nn.Sequential(
            nn.Linear(thermo_dim, 32),
            nn.GELU(),
            nn.Linear(32, desc_dim),
            nn.Sigmoid(),
        )

        # Descriptor head
        self.head = nn.Sequential(
            nn.Linear(desc_dim + thermo_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_properties),
        )

    def forward(self, descriptors, thermo_feat):
        """
        Args:
            descriptors: (B, 20) surface descriptors
            thermo_feat: (B, 5) temperature features [T, x1, 1/T, T², T³]
        """
        # Temperature-modulated descriptors
        gate = self.temp_embed(thermo_feat)
        modulated = descriptors * gate

        return self.head(torch.cat([modulated, thermo_feat], dim=-1))


class COSMOBridgeV51(nn.Module):
    """COSMOBridge v5.1: Better fusion + descriptor pathway + image-ready.

    Parameters
    ----------
    graph_dim : int
        Chemprop fingerprint dimension.
    surface_dim : int
        PointNet surface feature dimension.
    thermo_dim : int
        Full thermo+descriptor feature dimension (25).
    fused_dim : int
        Internal fusion dimension.
    n_properties : int
        Number of target properties.
    image_dim : int
        Image feature dimension (0 to disable image pathway).
    dropout : float
        Dropout probability.
    """

    def __init__(self, graph_dim=300, surface_dim=256, thermo_dim=25,
                 fused_dim=256, n_properties=7, image_dim=0, dropout=0.3):
        super().__init__()
        self.n_properties = n_properties
        self.use_images = image_dim > 0
        n_paths = 4 if self.use_images else 3

        # ── Path A: Cross-Attention Fusion (Graph × Surface) ──
        self.fusion = CrossAttentionFusion(
            graph_dim, surface_dim, fused_dim, n_heads=4, dropout=dropout)

        self.path_a_head = nn.Sequential(
            nn.Linear(fused_dim + thermo_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_properties),
        )

        # ── Path B: Chemprop Direct FFN (same as v4) ──
        self.path_b_head = nn.Sequential(
            nn.Linear(graph_dim + thermo_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_properties),
        )

        # ── Path C: Surface Descriptors + T-modulation (NEW) ──
        self.path_c = DescriptorPathway(
            desc_dim=20, thermo_dim=5, hidden=64,
            n_properties=n_properties, dropout=dropout)

        # ── Path D: Image (optional, for later) ──
        if self.use_images:
            self.path_d_head = nn.Sequential(
                nn.Linear(image_dim + thermo_dim, 64),
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(64, n_properties),
            )

        # ── Per-Molecule Router (extends v4 to 3-4 paths) ──
        router_input = graph_dim + surface_dim + thermo_dim
        if self.use_images:
            router_input += image_dim

        self.router = nn.Sequential(
            nn.Linear(router_input, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(64),
            nn.Linear(64, n_properties * n_paths),
        )

        # Initialize router: Path A and B dominant, C moderate, D low
        with torch.no_grad():
            self.router[-1].weight.zero_()
            bias = torch.zeros(n_properties * n_paths)
            for p in range(n_properties):
                # Per-property initialization based on gap analysis
                if n_paths == 3:
                    # [fusion, chemprop, descriptors]
                    if p in [0, 1]:  # gamma1, gamma2: fusion-heavy
                        bias[p*3:(p+1)*3] = torch.tensor([1.0, 0.5, -0.5])
                    elif p in [2, 4]:  # G_E, G_mix: fusion-heavy
                        bias[p*3:(p+1)*3] = torch.tensor([1.5, -0.5, 0.0])
                    elif p == 3:  # H_E: fusion + descriptors
                        bias[p*3:(p+1)*3] = torch.tensor([1.0, -0.5, 0.5])
                    elif p == 5:  # H_vap: descriptors dominate
                        bias[p*3:(p+1)*3] = torch.tensor([0.0, -0.5, 1.5])
                    elif p == 6:  # P: chemprop + descriptors
                        bias[p*3:(p+1)*3] = torch.tensor([0.0, 0.5, 1.0])
                else:
                    # [fusion, chemprop, descriptors, image]
                    bias[p*4:(p+1)*4] = torch.tensor([1.0, 0.0, 0.0, -2.0])
            self.router[-1].bias.copy_(bias)

        self.n_paths = n_paths

    def forward(self, graph_feat, surface_feat, thermo_feat, image_feat=None):
        """
        Args:
            graph_feat: (B, 300) frozen Chemprop fingerprint
            surface_feat: (B, 256) frozen PointNet features
            thermo_feat: (B, 25) [T, x1, 1/T, T², T³, desc_0..desc_19]
            image_feat: (B, image_dim) optional image features

        Returns:
            predictions: (B, 7)
            aux: dict with routing weights
        """
        # Split thermo into temperature (5D) and descriptors (20D)
        thermo_5 = thermo_feat[:, :5]
        descriptors = thermo_feat[:, 5:]

        # Path A: Cross-attention fusion
        fused = self.fusion(graph_feat, surface_feat)
        preds_a = self.path_a_head(torch.cat([fused, thermo_feat], dim=-1))

        # Path B: Chemprop direct
        preds_b = self.path_b_head(torch.cat([graph_feat, thermo_feat], dim=-1))

        # Path C: Descriptor pathway with T-modulation
        preds_c = self.path_c(descriptors, thermo_5)

        # Router
        router_input = torch.cat([graph_feat, surface_feat, thermo_feat], dim=-1)
        if self.use_images and image_feat is not None:
            router_input = torch.cat([router_input, image_feat], dim=-1)

        logits = self.router(router_input)
        logits = logits.view(-1, self.n_properties, self.n_paths)
        weights = F.softmax(logits, dim=-1)  # (B, 7, n_paths)

        # Stack predictions
        if self.use_images and image_feat is not None:
            preds_d = self.path_d_head(torch.cat([image_feat, thermo_feat], dim=-1))
            paths = torch.stack([preds_a, preds_b, preds_c, preds_d], dim=-1)
        else:
            paths = torch.stack([preds_a, preds_b, preds_c], dim=-1)

        # Weighted combination
        predictions = (paths * weights).sum(dim=-1)  # (B, 7)

        return predictions, {
            "weights": weights.detach(),
            "preds_a": preds_a.detach(),
            "preds_b": preds_b.detach(),
            "preds_c": preds_c.detach(),
        }

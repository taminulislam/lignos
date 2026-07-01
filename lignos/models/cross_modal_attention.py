"""Cross-Modal Attention Fusion for COSMOBridge v5.

Replaces the gated fusion from v3/v4 with attention-based fusion that allows
modalities to inform each other before the final per-property routing.

Cross-attention pairs:
  - ViT image <-> Graph (image features enriched by molecular structure)
  - ViT image <-> PointCloud (image features enriched by 3D surface geometry)
  - Siamese <-> Graph (ion interaction features enriched by bonding info)
  - Graph <-> PointCloud (inherited concept from v4)
"""

import torch
import torch.nn as nn


class CrossModalAttentionBlock(nn.Module):
    """Single cross-attention block: query attends to context.

    Includes residual connection, LayerNorm, and FFN.

    Parameters
    ----------
    dim : int
        Embedding dimension for both query and context.
    n_heads : int
        Number of attention heads.
    dropout : float
        Dropout probability.
    """

    def __init__(self, dim, n_heads=4, dropout=0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True
        )
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, query, context):
        """
        Args:
            query: (B, D) or (B, S_q, D) - query modality
            context: (B, D) or (B, S_kv, D) - context modality

        Returns:
            output: same shape as query, enriched by context
            weights: attention weights
        """
        # Handle 2D inputs by unsqueezing
        squeeze_q = query.dim() == 2
        if squeeze_q:
            query = query.unsqueeze(1)
        if context.dim() == 2:
            context = context.unsqueeze(1)

        # Pre-norm cross-attention
        q = self.norm_q(query)
        kv = self.norm_kv(context)
        attended, weights = self.attn(q, kv, kv)
        x = query + attended

        # FFN with residual
        x = x + self.ffn(self.norm_ffn(x))

        if squeeze_q:
            x = x.squeeze(1)

        return x, weights


class CrossModalFusion(nn.Module):
    """Cross-modal attention fusion across all modality pairs.

    Parameters
    ----------
    dim : int
        Common embedding dimension (all modalities projected to this).
    n_heads : int
        Number of attention heads per cross-attention block.
    n_modalities : int
        Number of input modalities.
    n_properties : int
        Number of target properties for per-property routing.
    dropout : float
        Dropout probability.
    """

    def __init__(self, dim=256, n_heads=4, n_modalities=5, n_properties=7,
                 dropout=0.1):
        super().__init__()
        self.dim = dim
        self.n_modalities = n_modalities
        self.n_properties = n_properties

        # Cross-attention pairs (key domain interactions)
        # ViT <-> Graph
        self.vit_graph = CrossModalAttentionBlock(dim, n_heads, dropout)
        # ViT <-> PointCloud
        self.vit_surface = CrossModalAttentionBlock(dim, n_heads, dropout)
        # Siamese <-> Graph
        self.siamese_graph = CrossModalAttentionBlock(dim, n_heads, dropout)
        # Graph <-> PointCloud
        self.graph_surface = CrossModalAttentionBlock(dim, n_heads, dropout)

        # Post-attention layer norms
        self.modality_norms = nn.ModuleList([
            nn.LayerNorm(dim) for _ in range(n_modalities)
        ])

        # Per-property modality routing weights
        # Shape: (n_properties, n_modalities) -> softmax -> weighted sum
        self.routing_logits = nn.Parameter(
            torch.zeros(n_properties, n_modalities)
        )

    def init_routing_from_domain_knowledge(self, init_values):
        """Initialize routing with domain knowledge.

        Args:
            init_values: dict mapping property name to list of 5 logit values,
                         or tensor of shape (n_properties, n_modalities).
        """
        if isinstance(init_values, dict):
            property_order = [
                "gamma1", "gamma2", "G_E", "H_E", "G_mix", "H_vap", "P"
            ]
            tensor = torch.zeros(self.n_properties, self.n_modalities)
            for i, prop in enumerate(property_order):
                if prop in init_values:
                    tensor[i] = torch.tensor(init_values[prop])
            init_values = tensor

        with torch.no_grad():
            self.routing_logits.copy_(init_values)

    def forward(self, modalities):
        """
        Args:
            modalities: list of 5 tensors, each (B, dim):
                [vit_emb, siamese_emb, graph_emb, surface_emb, tabular_emb]

        Returns:
            per_property_fused: (B, n_properties, dim)
                Fused representation for each property.
            routing_weights: (n_properties, n_modalities)
                Softmax routing weights (interpretable).
            cross_attn_aux: dict with attention weights for analysis.
        """
        m_vit, m_siam, m_graph, m_surface, m_tab = modalities
        aux = {}

        # Cross-modal attention (enrich modalities with each other)
        m_vit_g, aux["vit_graph_w"] = self.vit_graph(m_vit, m_graph)
        m_vit = m_vit + m_vit_g if m_vit_g.dim() == m_vit.dim() else m_vit + m_vit_g

        m_vit_s, aux["vit_surface_w"] = self.vit_surface(m_vit, m_surface)
        m_vit = m_vit + m_vit_s if m_vit_s.dim() == m_vit.dim() else m_vit + m_vit_s

        m_siam_g, aux["siamese_graph_w"] = self.siamese_graph(m_siam, m_graph)
        m_siam = m_siam + m_siam_g if m_siam_g.dim() == m_siam.dim() else m_siam + m_siam_g

        m_graph_s, aux["graph_surface_w"] = self.graph_surface(m_graph, m_surface)
        m_graph = m_graph + m_graph_s if m_graph_s.dim() == m_graph.dim() else m_graph + m_graph_s

        # Normalize all modalities
        normed = [
            self.modality_norms[0](m_vit),
            self.modality_norms[1](m_siam),
            self.modality_norms[2](m_graph),
            self.modality_norms[3](m_surface),
            self.modality_norms[4](m_tab),
        ]
        stacked = torch.stack(normed, dim=1)  # (B, 5, dim)

        # Per-property weighted fusion
        routing_weights = torch.softmax(self.routing_logits, dim=-1)  # (P, 5)

        # (B, 5, dim) x (P, 5) -> (B, P, dim)
        # For each property, weighted sum across modalities
        per_property_fused = torch.einsum(
            "bmd, pm -> bpd", stacked, routing_weights
        )

        return per_property_fused, routing_weights.detach(), aux

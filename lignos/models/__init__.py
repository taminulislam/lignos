"""COSMOBridge v5: Full Multimodal Architecture with Image Superpowers.

Models:
    - MultiViewViT: 36-view ViT-Tiny with view-level self-attention
    - CationAnionSiamese: Siamese encoder with cross-attention for ion pairs
    - CrossModalFusion: Attention-based fusion with per-property routing
    - SimCLR: Contrastive pre-training wrapper
    - COSMOBridgeV5: Full unified architecture
"""

from .multiview_vit import MultiViewViT
from .siamese_encoder import CationAnionSiamese
from .cross_modal_attention import CrossModalFusion, CrossModalAttentionBlock
from .simclr import SimCLR
from .cosmobridge_v5 import COSMOBridgeV5

__all__ = [
    "MultiViewViT",
    "CationAnionSiamese",
    "CrossModalFusion",
    "CrossModalAttentionBlock",
    "SimCLR",
    "COSMOBridgeV5",
]

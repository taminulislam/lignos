"""SimCLR Contrastive Pre-training for COSMO Surface Images.

Self-supervised pre-training on 1,500+ IL COSMO surfaces. Uses the natural
36 rotation views per molecule as positive pairs (no synthetic augmentation
needed for pair generation).

Training recipe:
    - Positive pairs: two random rotation views of the same molecule
    - Negative pairs: views from different molecules in the batch
    - Loss: NT-Xent (normalized temperature-scaled cross-entropy)
    - Encoder: ViT-Tiny (shared weights)
    - Projection head: 2-layer MLP (discarded after pre-training)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimCLR(nn.Module):
    """SimCLR wrapper around any encoder.

    Parameters
    ----------
    encoder : nn.Module
        Backbone encoder (e.g., ViT-Tiny) that maps (B, C, H, W) -> (B, D).
    encoder_dim : int
        Output dimension of the encoder.
    projection_dim : int
        Dimension of the projection head output (contrastive space).
    projection_hidden : int
        Hidden dimension in the projection MLP.
    temperature : float
        Temperature parameter for NT-Xent loss.
    """

    def __init__(
        self,
        encoder,
        encoder_dim=192,
        projection_dim=128,
        projection_hidden=256,
        temperature=0.07,
    ):
        super().__init__()
        self.encoder = encoder
        self.temperature = temperature

        # Projection head (discarded after pre-training)
        self.projection = nn.Sequential(
            nn.Linear(encoder_dim, projection_hidden),
            nn.BatchNorm1d(projection_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(projection_hidden, projection_dim),
        )

    def encode(self, x):
        """Encode images to representation space (before projection).

        Args:
            x: (B, C, H, W) input images

        Returns:
            h: (B, encoder_dim) representations
        """
        return self.encoder(x)

    def project(self, h):
        """Project representations to contrastive space.

        Args:
            h: (B, encoder_dim) encoder output

        Returns:
            z: (B, projection_dim) normalized projections
        """
        z = self.projection(h)
        return F.normalize(z, dim=1)

    def nt_xent_loss(self, z_i, z_j):
        """Compute NT-Xent loss for a batch of positive pairs.

        Args:
            z_i: (B, D) projections from view 1
            z_j: (B, D) projections from view 2

        Returns:
            loss: scalar NT-Xent loss
        """
        B = z_i.shape[0]
        z = torch.cat([z_i, z_j], dim=0)  # (2B, D)

        # Cosine similarity matrix
        sim = torch.mm(z, z.t()) / self.temperature  # (2B, 2B)

        # Mask out self-similarity on diagonal
        mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, float("-inf"))

        # Positive pairs: (i, i+B) and (i+B, i)
        labels = torch.cat([
            torch.arange(B, 2 * B, device=z.device),
            torch.arange(B, device=z.device),
        ])

        return F.cross_entropy(sim, labels)

    def forward(self, view1, view2):
        """
        Args:
            view1: (B, C, H, W) - first rotation view
            view2: (B, C, H, W) - second rotation view

        Returns:
            loss: NT-Xent contrastive loss
            aux: dict with embeddings for monitoring
        """
        h1 = self.encode(view1)
        h2 = self.encode(view2)

        z1 = self.project(h1)
        z2 = self.project(h2)

        loss = self.nt_xent_loss(z1, z2)

        aux = {
            "h1": h1.detach(),
            "h2": h2.detach(),
            "z1": z1.detach(),
            "z2": z2.detach(),
        }

        return loss, aux


class COSMOViewPairDataset(torch.utils.data.Dataset):
    """Dataset that yields random pairs of rotation views per molecule.

    Parameters
    ----------
    image_dir : str
        Root directory containing {mol_id}_frames/ subdirectories.
    mol_ids : list[str]
        List of molecule IDs to include.
    transform : callable, optional
        Image transforms (augmentations).
    n_views : int
        Total number of rotation views per molecule.
    """

    def __init__(self, image_dir, mol_ids, transform=None, n_views=36):
        from pathlib import Path
        self.image_dir = Path(image_dir)
        self.mol_ids = mol_ids
        self.transform = transform
        self.n_views = n_views

    def __len__(self):
        return len(self.mol_ids)

    def __getitem__(self, idx):
        from PIL import Image
        import random

        mol_id = self.mol_ids[idx]
        frame_dir = self.image_dir / f"{mol_id}_frames"

        # Pick two random rotation views
        i, j = random.sample(range(self.n_views), 2)
        path1 = frame_dir / f"frame_{i:03d}.png"
        path2 = frame_dir / f"frame_{j:03d}.png"

        img1 = Image.open(path1).convert("RGB")
        img2 = Image.open(path2).convert("RGB")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, mol_id

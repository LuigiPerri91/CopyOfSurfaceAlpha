"""
Vision Transformer encoder for the implied volatility surface.

Input:  (batch, C=6, H=12, W=20)
Output: (batch, embed_dim=128)  — z_surf

Grid is split into (patch_h=3, patch_w=4) patches → 20 patches total.
A learnable CLS token attends to all patches. CLS output = z_surf.
"""

import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, num_channels: int, patch_size_h: int, patch_size_w: int, embed_dim: int, grid_h: int, grid_w: int):
        super().__init__()
        self.ph = patch_size_h
        self.pw = patch_size_w
        self.nh = grid_h // patch_size_h
        self.nw = grid_w // patch_size_w
        self.num_patches = self.nh * self.nw
        self.projection = nn.Linear(num_channels * patch_size_h * patch_size_w, embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B,C,H,W)
        B,C,H, W =x.shape
        x = x.reshape(B,C,self.nh, self.ph, self.nw, self.pw)
        x = x.permute(0,2,4,1,3,5) # (B, nh, nw, C, ph, pw)
        x = x.reshape(B, self.num_patches, C * self.ph * self.pw)
        return self.projection(x) # (B, num_patches, embed_dim)

class SurfaceEncoder(nn.Module):
    """
    ViT encoder for the IV surface.
    PatchEmbedding -> CLS token -> position embed -> N x TransformerEncoderLayer -> z_surf
    """
    def __init__(
        self,
        num_input_channels: int = 6,
        patch_size_h: int = 3,
        patch_size_w: int = 4,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,  # applied to both attention weights and feedforward sublayer
        grid_h: int = 12,
        grid_w: int = 20
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(num_input_channels, patch_size_h, patch_size_w, embed_dim, grid_h, grid_w)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1,num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # pre-norm for stable training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, surface: torch.Tensor) -> torch.Tensor:
        """
        Args:
            surface: (batch, 6, H, W)
        Returns:
            z_surf: (batch, embed_dim)
        """
        B = surface.shape[0] 
        x = self.patch_embed(surface) # (B, num_patches, embed_dim)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim =1) # (B, num_patches+1, embed_dim)
        x = self.pos_drop(x + self.pos_embed)
        x = self.norm(self.transformer(x))
        return x[:,0] # CLS token → (B, embed_dim)
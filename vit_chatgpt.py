import torch
import torch.nn as nn
import math


# ----------------------------
# Patch Embedding
# ----------------------------
class PatchEmbedding(nn.Module):
    """
    Image -> Patch tokens
    """
    def __init__(self, img_size=32, patch_size=8, in_ch=3, embed_dim=64):
        super().__init__()
        assert img_size % patch_size == 0
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels=in_ch,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        """
        x: (B, C, H, W)
        returns: (B, N, D)
        """
        x = self.proj(x)          # (B, D, H', W')
        x = x.flatten(2)          # (B, D, N)
        x = x.transpose(1, 2)     # (B, N, D)
        return x


# ----------------------------
# Multi-Head Self Attention
# ----------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        """
        x: (B, N, D)
        """
        B, N, D = x.shape

        qkv = self.qkv(x)                       # (B, N, 3D)
        qkv = qkv.view(B, N, 3, self.num_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)        # (3, B, H, N, d_k)

        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = torch.softmax(attn, dim=-1)

        out = attn @ v                          # (B, H, N, d_k)
        out = out.transpose(1, 2).contiguous()
        out = out.view(B, N, D)

        return self.out(out)


# ----------------------------
# Transformer Encoder Block
# ----------------------------
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Linear(d_model * mlp_ratio, d_model)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ----------------------------
# ViT (minimal)
# ----------------------------
class MiniViT(nn.Module):
    def __init__(self, img_size=32, patch_size=8, embed_dim=64, num_heads=4):
        super().__init__()
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim
        )

        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        self.block = TransformerBlock(embed_dim, num_heads)

    def forward(self, x):
        x = self.patch_embed(x)      # (B, N, D)
        x = x + self.pos_embed       # positional encoding
        x = self.block(x)
        return x


# ----------------------------
# Test / main
# ----------------------------
def main():
    B, C, H, W = 2, 3, 32, 32
    x = torch.randn(B, C, H, W)

    model = MiniViT(
        img_size=32,
        patch_size=8,
        embed_dim=64,
        num_heads=4
    )

    out = model(x)

    print("Input shape :", x.shape)
    print("Output shape:", out.shape)
    # Expected: (B, N, D) = (2, 16, 64)


if __name__ == "__main__":
    main()

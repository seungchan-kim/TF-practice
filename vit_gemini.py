import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        # TODO: 이미지를 패치로 나누고 프로젝션하는 Conv2d 레이어 정의
        # kernel_size와 stride를 patch_size와 동일하게 설정하는 것이 핵심
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # x: [B, 3, H, W]
        x = self.proj(x) # [B, embed_dim, H/P, W/P]
        x = x.flatten(2) # [B, embed_dim, L] -> 여기서 L = (H/P * W/P)
        x = x.transpose(1, 2) # [B, L, embed_dim] -> (Batch, Seq_Len, Dimension)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, L, D = x.shape
        # qkv: [B, L, 3*D] -> [B, L, 3, num_heads, head_dim] -> [3, B, num_heads, L, head_dim]
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # TODO: Attention score 계산 (q @ k.T)
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        
        # TODO: weighted sum (attn @ v)
        x = (attn @ v).transpose(1, 2).reshape(B, L, D)
        x = self.proj(x)
        return x

class ViTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        )

    def forward(self, x):
        # Residual Connection (Pre-Norm style)
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class SimpleViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=768, num_heads=12, num_layers=6):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        
        # TODO: CLS Token 및 Position Embedding 정의
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        self.blocks = nn.ModuleList([
            ViTBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # 1. Image to Patches
        x = self.patch_embed(x) # [B, L, D]
        
        # 2. Add CLS Token
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1) # [B, L+1, D]
        
        # 3. Add Positional Embedding
        x = x + self.pos_embed
        
        # 4. Transformer Blocks
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        return x # CLS token output: x[:, 0]

# --- Main Test Function ---
if __name__ == "__main__":
    # 데이터 준비: (Batch, Channels, Height, Width)
    dummy_img = torch.randn(2, 3, 224, 224) 
    
    # 모델 초기화
    model = SimpleViT(img_size=224, patch_size=16, embed_dim=128, num_heads=4, num_layers=2)
    
    print(f"Input image shape: {dummy_img.shape}")
    
    # Forward pass
    output = model(dummy_img)
    
    print(f"Output sequence shape: {output.shape}") # [B, L+1, D]
    
    # 197 = (224/16)^2 + 1 (cls token)
    assert output.shape == (2, 197, 128), "Shape mismatch!"
    print("\n[Success] Vision Transformer pipeline is working correctly!")
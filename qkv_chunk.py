import math
import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention.

    Input:
        x: (B, T, D)
        attn_mask: None or (B, T)
    Output:
        (B, T, D)
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # projections
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, attn_mask=None):
        B, T, D = x.shape

        # TODO 1: project x to q, k, v
        # q, k, v: (B, T, D)
        qkv = self.qkv_proj(x)
        q,k,v = qkv.chunk(3,dim=-1)

        # TODO 2: reshape q, k, v for multi-head attention
        # target shape: (B, num_heads, T, head_dim)
        q = q.reshape(B,T,self.num_heads, self.head_dim).transpose(1,2)
        k = k.reshape(B,T,self.num_heads, self.head_dim).transpose(1,2)
        v = v.reshape(B,T,self.num_heads, self.head_dim).transpose(1,2)

        # TODO 3: compute scaled dot-product attention scores
        # scores: (B, num_heads, T, T)
        scores = torch.matmul(q,k.transpose(2,3)) / math.sqrt(self.head_dim)

        # TODO 4: apply attention mask if provided
        if attn_mask is not None:
            # attn_mask: (B, T)
            # broadcast appropriately
            mask = attn_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # TODO 5: softmax
        attn = torch.softmax(scores, dim=-1)

        # TODO 6: weighted sum of values
        # out: (B, num_heads, T, head_dim)
        out = torch.matmul(attn, v)

        # TODO 7: merge heads back
        # out: (B, T, D)
        out = (out.transpose(1,2).contiguous().view(B,T,D))

        # TODO 8: final projection
        out = self.out_proj(out)

        return out


# -------------------------
# Sanity check
# -------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    B, T, D = 2, 4, 8 #batch size, T=sequence len (number of tokens); D= dimension
    H = 2

    x = torch.randn(B, T, D)
    mask = torch.tensor([[1, 1, 1, 0],
                         [1, 1, 0, 0]])

    attn = MultiHeadSelfAttention(D, H)
    y = attn(x, mask)

    print("Output shape:", y.shape)  # expect (B, T, D)

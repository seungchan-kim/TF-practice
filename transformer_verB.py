"""
TRANSFORMER IMPLEMENTATION - VERSION B (Minimal Interview Style)
================================================================
Time Limit: 45 minutes
Difficulty: ⭐⭐⭐☆☆

INTERVIEW SCENARIO:
"We need you to implement a Transformer encoder from scratch using PyTorch.
Focus on correctness first, optimization later. Here's the starter code."

REQUIREMENTS:
✓ Multi-head self-attention with scaled dot-product
✓ Position-wise feed-forward network
✓ Sinusoidal positional encoding
✓ Layer normalization with residual connections
✓ Supports batched input

EVALUATION CRITERIA:
- Does it run without errors?
- Correct output shapes?
- Clean, readable code?
- Handles edge cases?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # TODO: Initialize projection layers (Q, K, V, Output)
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query, key, value: (batch, seq_len, d_model)
            mask: (batch, 1, 1, seq_len) or (batch, 1, seq_len, seq_len)
        Returns:
            output: (batch, seq_len, d_model)
        """
        # TODO: Implement multi-head attention
        #from pdb import set_trace as bp; bp()
        q_proj = self.Wq(query)#torch.matmul(query, self.Wq) #query is B x L x d_model, d_modelxd_model => B x L x d_model
        k_proj = self.Wk(key) #torch.matmul(key, self.Wk)
        v_proj = self.Wv(value) #torch.matmul(value, self.Wv)

        #split heads
        q_split = self.split_heads(q_proj) #B x num_heads x seq_len x d_k
        k_split = self.split_heads(k_proj)
        v_split = self.split_heads(v_proj)

        #multi-head attention
        scores = self.scaled_dot_product_attention(q_split, k_split, v_split, mask)
        combined = self.combine_heads(scores)
        output = self.Wo(combined)
        return output

    def scaled_dot_product_attention(self, q,k,v,mask=None):
        #q,k,v shape is B x num_heads x seq_len x d_k
        scores = torch.matmul(q,k.transpose(2,3))/math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights , v)
        return output


    def combine_heads(self, x):
        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(1,2).contiguous().view(batch_size, seq_len, self.d_model)
        return x

    def split_heads(self, x):
        #x is batch_size x len x d_model
        #d_model -> split into num_heads and d_k
        batch_size, seq_len = x.shape[0], x.shape[1]
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        x = x.transpose(1,2)
        return x



        


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        # TODO: Initialize layers
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        # TODO: Implement feed-forward network
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # TODO: Create positional encoding matrix
        # Use sinusoidal functions as described in "Attention is All You Need"

        pe = torch.zeros(max_len, d_model)
        
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2) * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.encoding = pe
    
    def forward(self, x):
        """
        Add positional encoding to input
        
        Args:
            x: shape (batch, seq_len, d_model)
        
        Returns:
            x + positional encoding
        """
        # TODO: Add positional encoding to input
        seq_len = x.shape[1]
        # Add positional encoding (broadcasting over batch dimension)
        return x + self.encoding[:seq_len, :]


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        # TODO: Initialize sublayers
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: optional attention mask
        Returns:
            output: (batch, seq_len, d_model)
        """
        # TODO: Implement encoder layer with:
        # 1. Multi-head self-attention + residual + norm
        # 2. Feed-forward + residual + norm
        attn_output = self.attention.forward(x,x,x,mask)
        x = self.norm1(x+attn_output)
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6, 
                 d_ff=2048, max_len=5000, dropout=0.1):
        super().__init__()
        # TODO: Initialize all components
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
    
    def forward(self, x, mask=None):
        """
        Args:
            x: token indices (batch, seq_len)
            mask: optional attention mask
        Returns:
            output: (batch, seq_len, d_model)
        """
        # TODO: Implement full forward pass
        x = self.embedding(x) #->batch_size x seq_len => batch_size x seq_len x d_model
        x = self.pos_encoding.forward(x)
        for layer in self.layers:
            x = layer.forward(x, mask)
        return x



# ============================================================================
# TEST SUITE
# ============================================================================

def test_basic():
    """Basic functionality test"""
    print("Test 1: Basic Forward Pass")
    print("-" * 50)
    
    batch_size = 2
    seq_len = 10
    vocab_size = 1000
    d_model = 512
    
    model = TransformerEncoder(vocab_size, d_model=d_model, num_layers=2)
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    try:
        output = model(x)
        assert output.shape == (batch_size, seq_len, d_model)
        print(f"✓ Output shape correct: {output.shape}")
        print(f"✓ Test passed!\n")
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}\n")
        return False


def test_different_seq_lengths():
    """Test with different sequence lengths"""
    print("Test 2: Variable Sequence Lengths")
    print("-" * 50)
    
    vocab_size = 1000
    d_model = 256
    model = TransformerEncoder(vocab_size, d_model=d_model, num_layers=1)
    
    try:
        for seq_len in [5, 10, 20, 50]:
            x = torch.randint(0, vocab_size, (2, seq_len))
            output = model(x)
            assert output.shape == (2, seq_len, d_model)
        print("✓ Handles variable sequence lengths")
        print(f"✓ Test passed!\n")
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}\n")
        return False


def test_gradient_flow():
    """Test if gradients flow correctly"""
    print("Test 3: Gradient Flow")
    print("-" * 50)
    
    vocab_size = 1000
    d_model = 128
    model = TransformerEncoder(vocab_size, d_model=d_model, num_layers=1)
    x = torch.randint(0, vocab_size, (2, 10))
    
    try:
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        # Check if gradients exist
        has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        assert has_grad, "No gradients computed!"
        
        print("✓ Gradients computed successfully")
        print(f"✓ Test passed!\n")
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}\n")
        return False


def test_attention_mask():
    """Test attention masking (padding mask)"""
    print("Test 4: Attention Masking")
    print("-" * 50)
    
    vocab_size = 1000
    d_model = 128
    batch_size = 2
    seq_len = 10
    
    model = TransformerEncoder(vocab_size, d_model=d_model, num_layers=1)
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Create padding mask (first sequence has padding at positions 5-9)
    mask = torch.ones(batch_size, 1, 1, seq_len)
    mask[0, 0, 0, 5:] = 0  # Mask out positions 5-9 for first sequence
    
    try:
        output = model(x, mask=mask)
        assert output.shape == (batch_size, seq_len, d_model)
        print("✓ Attention mask applied successfully")
        print(f"✓ Test passed!\n")
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}\n")
        return False


def test_numerical_stability():
    """Check for NaN/Inf in outputs"""
    print("Test 5: Numerical Stability")
    print("-" * 50)
    
    vocab_size = 1000
    d_model = 256
    model = TransformerEncoder(vocab_size, d_model=d_model, num_layers=3)
    model.eval()
    
    x = torch.randint(0, vocab_size, (4, 20))
    
    try:
        with torch.no_grad():
            output = model(x)
        
        assert not torch.isnan(output).any(), "NaN detected in output!"
        assert not torch.isinf(output).any(), "Inf detected in output!"
        
        print(f"✓ No NaN/Inf in outputs")
        print(f"✓ Output mean: {output.mean().item():.4f}")
        print(f"✓ Output std: {output.std().item():.4f}")
        print(f"✓ Test passed!\n")
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}\n")
        return False


def run_all_tests():
    """Run complete test suite"""
    print("\n" + "=" * 60)
    print("TRANSFORMER ENCODER - TEST SUITE")
    print("=" * 60 + "\n")
    
    results = []
    results.append(("Basic Forward Pass", test_basic()))
    results.append(("Variable Seq Lengths", test_different_seq_lengths()))
    results.append(("Gradient Flow", test_gradient_flow()))
    results.append(("Attention Masking", test_attention_mask()))
    results.append(("Numerical Stability", test_numerical_stability()))
    
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8s} {name}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\nScore: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 Congratulations! All tests passed!")
        print("Your Transformer implementation is working correctly.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Keep debugging!")
    
    print("=" * 60)


def show_model_info():
    """Display model architecture and parameter count"""
    print("\n" + "=" * 60)
    print("MODEL INFORMATION")
    print("=" * 60)
    
    vocab_size = 10000
    d_model = 512
    model = TransformerEncoder(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=8,
        num_layers=6,
        d_ff=2048
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Configuration:")
    print(f"  - Vocabulary size: {vocab_size:,}")
    print(f"  - Model dimension: {d_model}")
    print(f"  - Number of heads: 8")
    print(f"  - Number of layers: 6")
    print(f"  - Feed-forward dim: 2048")
    print(f"\nParameters:")
    print(f"  - Total: {total_params:,}")
    print(f"  - Trainable: {trainable_params:,}")
    print(f"  - Size: ~{total_params * 4 / 1024 / 1024:.1f} MB (fp32)")
    print("=" * 60)


# ============================================================================
# QUICK REFERENCE
# ============================================================================

def show_formulas():
    """Show key formulas for reference"""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                     QUICK REFERENCE                       ║
    ╚═══════════════════════════════════════════════════════════╝
    
    ATTENTION:
    ┌─────────────────────────────────────────────────────────┐
    │ Attention(Q, K, V) = softmax(QK^T / √d_k) V             │
    │                                                         │
    │ MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O        │
    │   where head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)     │
    └─────────────────────────────────────────────────────────┘
    
    POSITIONAL ENCODING:
    ┌─────────────────────────────────────────────────────────┐
    │ PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))          │
    │ PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))          │
    └─────────────────────────────────────────────────────────┘
    
    FEED-FORWARD:
    ┌─────────────────────────────────────────────────────────┐
    │ FFN(x) = max(0, xW_1 + b_1)W_2 + b_2                    │
    └─────────────────────────────────────────────────────────┘
    
    LAYER NORMALIZATION:
    ┌─────────────────────────────────────────────────────────┐
    │ LayerNorm(x) = γ * (x - μ) / √(σ² + ε) + β             │
    └─────────────────────────────────────────────────────────┘
    
    RESIDUAL CONNECTION:
    ┌─────────────────────────────────────────────────────────┐
    │ output = LayerNorm(x + Sublayer(x))                     │
    └─────────────────────────────────────────────────────────┘
    """)


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║        TRANSFORMER ENCODER IMPLEMENTATION                ║
    ║              Version B - Minimal Style                   ║
    ╚═══════════════════════════════════════════════════════════╝
    
    Welcome to the coding interview!
    
    Your task: Implement a Transformer encoder from scratch.
    Time limit: 45 minutes
    
    Tips:
    • Start with the smallest compone nt (attention, FFN, etc.)
    • Test each component individually before integration
    • Use show_formulas() if you need to review the math
    • Run run_all_tests() to check your implementation
    
    Uncomment below to get started:
    """)
    
    # Uncomment these as you implement:
    #show_formulas()       # Show math formulas
    run_all_tests()       # Run all tests
    #show_model_info()     # See model stats
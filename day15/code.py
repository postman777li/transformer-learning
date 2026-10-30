import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. RMSNorm：不减均值，按 RMS 缩放
# ==========================================

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        # x: (B, T, C)
        rms = torch.sqrt(
            x.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        return (x / rms) * self.weight

# ==========================================
# 2. RoPE：根据位置旋转 Q、K
# ==========================================

def apply_rope(q, k, start_pos=0, theta=10000.0):
    # q: (B, Hq, T, D)
    # k: (B, Hkv, T, D)
    T = q.size(-2)
    D = q.size(-1)
    assert D % 2 == 0

    device = q.device

    # 每一对维度使用一个旋转频率
    inv_freq = 1.0 / (
        theta ** (
            torch.arange(0, D, 2, device=device).float() / D
        )
    )

    positions = torch.arange(
        start_pos, start_pos + T, device=device
    ).float()

    # (T, D/2)
    freqs = torch.outer(positions, inv_freq)

    # (T, D)，前后两半使用相同频率
    emb = torch.cat([freqs, freqs], dim=-1)

    cos = emb.cos()[None, None, :, :]
    sin = emb.sin()[None, None, :, :]

    def rotate_half(x):
        first, second = x.chunk(2, dim=-1)
        return torch.cat([-second, first], dim=-1)

    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin

    return q, k

# ==========================================
# 3. GQA：多个 Query Head 共享 K、V Head
# ==========================================

class GQAAttention(nn.Module):
    def __init__(
        self,
        d_model=128,
        num_heads=8,
        num_kv_heads=2
    ):
        super().__init__()

        assert d_model % num_heads == 0
        assert num_heads % num_kv_heads == 0

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.num_repeats = num_heads // num_kv_heads

        D = self.head_dim

        # Q 有 8 个 Head；K、V 各只有 2 个 Head
        self.q_proj = nn.Linear(
            d_model, num_heads * D, bias=False
        )
        self.k_proj = nn.Linear(
            d_model, num_kv_heads * D, bias=False
        )
        self.v_proj = nn.Linear(
            d_model, num_kv_heads * D, bias=False
        )
        self.o_proj = nn.Linear(
            d_model, d_model, bias=False
        )

    def forward(self, x):
        B, T, C = x.shape
        H, HKV, D = (
            self.num_heads,
            self.num_kv_heads,
            self.head_dim
        )

        # 投影并拆分 Head
        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, T, HKV, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, HKV, D).transpose(1, 2)

        # RoPE 作用于 Q、K，不作用于 V
        q, k = apply_rope(q, k)

        # 为了进行矩阵计算，把 K、V 分组扩展到 8 个 Head
        # (B, 2, T, D) -> (B, 8, T, D)
        k = k.repeat_interleave(self.num_repeats, dim=1)
        v = v.repeat_interleave(self.num_repeats, dim=1)

        # Attention Score: (B, H, T, T)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(D)

        # GPT 因果 Mask：屏蔽未来 Token
        mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device),
            diagonal=1
        )
        scores = scores.masked_fill(mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)

        # (B, H, T, T) @ (B, H, T, D)
        # -> (B, H, T, D)
        out = weights @ v

        # 合并 Head -> (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.o_proj(out)

# ==========================================
# 4. SwiGLU：门控式 FFN
# ==========================================

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()

        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        features = self.up_proj(x)

        # 两条分支逐元素相乘，再投影回原维度
        return self.down_proj(gate * features)

# ==========================================
# 5. Llama 风格 Transformer Block
# ==========================================

class LlamaBlock(nn.Module):
    def __init__(
        self,
        d_model=128,
        num_heads=8,
        num_kv_heads=2,
        d_ff=352
    ):
        super().__init__()

        self.attn_norm = RMSNorm(d_model)
        self.attn = GQAAttention(
            d_model, num_heads, num_kv_heads
        )

        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x):
        # Pre-Norm + Attention 残差连接
        x = x + self.attn(self.attn_norm(x))

        # Pre-Norm + SwiGLU 残差连接
        x = x + self.ffn(self.ffn_norm(x))

        return x

# ==========================================
# 6. 运行验证
# ==========================================

torch.manual_seed(42)

# 两个句子，每句 10 个 Token，每个 Token 128 维
x = torch.randn(2, 10, 128)

block = LlamaBlock(
    d_model=128,
    num_heads=8,
    num_kv_heads=2,
    d_ff=352
)

y = block(x)

print("Input shape:", x.shape)
print("Output shape:", y.shape)
print(
    "Parameters:",
    sum(p.numel() for p in block.parameters())
)

# 验证反向传播
loss = y.square().mean()
loss.backward()

print(
    "Gradient check:",
    block.attn.q_proj.weight.grad is not None
)

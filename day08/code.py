
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()

        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # 一次投影生成 Q、K、V
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.shape
        H = self.num_heads
        D = self.head_dim

        # (B, T, C) -> (B, T, 3*C)
        qkv = self.qkv_proj(x)

        # (B, T, 3*C) -> (B, T, 3, H, D)
        qkv = qkv.view(B, T, 3, H, D)

        # -> (3, B, H, T, D)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # (B, H, T, D) @ (B, H, D, T)
        # -> (B, H, T, T)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(D)

        # True 表示禁止关注未来位置
        future_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device),
            diagonal=1
        )

        scores = scores.masked_fill(future_mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)

        # (B, H, T, T) @ (B, H, T, D)
        # -> (B, H, T, D)
        out = weights @ v

        # 合并多个 Head -> (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(out), weights


class FeedForward(nn.Module):
    def __init__(self, d_model=512, d_ff=2048):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=2048):
        super().__init__()

        # Pre-LN：先归一化，再进入子层
        self.ln1 = nn.LayerNorm(d_model)
        self.attention = CausalSelfAttention(d_model, num_heads)

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(self, x):
        # 第一条残差连接：Attention
        attn_out, weights = self.attention(self.ln1(x))
        x = x + attn_out

        # 第二条残差连接：FFN
        ffn_out = self.ffn(self.ln2(x))
        x = x + ffn_out

        return x, weights


# ==================== 验证 Block ====================

torch.manual_seed(42)

B, T, d_model = 2, 10, 512

# 模拟两个句子的输入表示
x = torch.randn(B, T, d_model)

block = TransformerBlock(
    d_model=512,
    num_heads=8,
    d_ff=2048
)

y, weights = block(x)

print("Input:", x.shape)
print("Output:", y.shape)
print("Attention weights:", weights.shape)

# 验证因果约束
future_mask = torch.triu(
    torch.ones(T, T, dtype=torch.bool), diagonal=1
)
assert torch.all(weights.masked_select(
    future_mask.unsqueeze(0).unsqueeze(0).expand_as(weights)
) == 0)

# 验证反向传播能够更新可学习参数
loss = y.square().mean()
loss.backward()

assert block.attention.qkv_proj.weight.grad is not None

print("Causal mask check passed.")
print("Gradient check passed.")

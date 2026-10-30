
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
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniGPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=512,
        num_heads=8,
        num_layers=4,
        max_len=128
    ):
        super().__init__()

        self.max_len = max_len

        # Token Embedding
        self.token_emb = nn.Embedding(
            vocab_size, d_model
        )

        # 可学习位置 Embedding
        self.pos_emb = nn.Embedding(
            max_len, d_model
        )

        # 多个 Transformer Block
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads)
            for _ in range(num_layers)
        ])

        # 最终归一化
        self.ln_final = nn.LayerNorm(d_model)

        # LM Head：隐藏向量映射到词表
        self.lm_head = nn.Linear(
            d_model, vocab_size, bias=False
        )

    def forward(self, input_ids):
        B, T = input_ids.shape

        assert T <= self.max_len

        # ① Token Embedding
        token_vectors = self.token_emb(input_ids)

        # ② Position Embedding
        positions = torch.arange(
            T, device=input_ids.device
        )

        position_vectors = self.pos_emb(positions)

        # ③ 融合 Token 和 Position 信息
        x = token_vectors + position_vectors

        # ④ 经过多个 Transformer Block
        for block in self.blocks:
            x, _ = block(x)

        # ⑤ 最终归一化
        x = self.ln_final(x)

        # ⑥ 映射到词表，得到 Logits
        logits = self.lm_head(x)

        return logits


torch.manual_seed(42)

vocab_size = 10000

model = MiniGPT(
    vocab_size=vocab_size,
    d_model=512,
    num_heads=8,
    num_layers=4,
    max_len=128
)

# 两个序列，每个序列10个Token
input_ids = torch.randint(
    0, vocab_size, (2, 10)
)

logits = model(input_ids)

print("Input:", input_ids.shape)
print("Logits:", logits.shape)

# 取第一个序列最后一个位置的预测
last_logits = logits[0, -1, :]

probs = F.softmax(last_logits, dim=-1)

predicted_id = torch.argmax(probs).item()

print("Predicted Token ID:", predicted_id)

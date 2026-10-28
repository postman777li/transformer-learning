
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
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model=512, max_len=512):
        super().__init__()

        assert d_model % 2 == 0

        position = torch.arange(
            max_len, dtype=torch.float32
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        T = x.size(1)
        return x + self.pe[:, :T, :]


class MiniGPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=512,
        num_heads=8,
        num_layers=4,
        d_ff=2048,
        max_len=512
    ):
        super().__init__()

        # Token ID → 向量
        self.token_embedding = nn.Embedding(
            vocab_size, d_model
        )

        # 固定正弦位置编码
        self.position = PositionalEncoding(
            d_model, max_len
        )

        # 堆叠多个 Day 8 的 TransformerBlock
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

        # 512 维 → 整个词表大小
        self.lm_head = nn.Linear(
            d_model, vocab_size, bias=False
        )

    def forward(self, input_ids):
        # input_ids: (B, T)

        # ① Embedding
        x = self.token_embedding(input_ids)

        # ② 位置编码
        x = self.position(x)

        # ③ 逐层经过 TransformerBlock
        for block in self.blocks:
            x, _ = block(x)

        # ④ 最终归一化
        x = self.final_norm(x)

        # ⑤ 映射成词表 logits
        logits = self.lm_head(x)

        # (B, T, vocab_size)
        return logits

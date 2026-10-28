
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# ==================================================
# 1. 通用 Multi-Head Attention
# ==================================================

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        B, T, C = x.shape
        return x.view(
            B, T, self.num_heads, self.head_dim
        ).transpose(1, 2)

    def forward(self, query, key_value, causal=False):
        # query:     (B, Tq, C)
        # key_value: (B, Tk, C)

        B, Tq, C = query.shape
        Tk = key_value.size(1)

        q = self.split_heads(self.wq(query))
        k = self.split_heads(self.wk(key_value))
        v = self.split_heads(self.wv(key_value))

        # scores: (B, H, Tq, Tk)
        scores = (
            q @ k.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        if causal:
            assert Tq == Tk
            # True 表示禁止关注未来位置
            future_mask = torch.triu(
                torch.ones(
                    Tq, Tk, dtype=torch.bool,
                    device=query.device
                ),
                diagonal=1
            )
            scores = scores.masked_fill(
                future_mask, float("-inf")
            )

        weights = F.softmax(scores, dim=-1)

        out = weights @ v
        out = out.transpose(1, 2).contiguous()
        out = out.view(B, Tq, C)

        return self.wo(out), weights


# ==================================================
# 2. FFN
# ==================================================

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


# ==================================================
# 3. Encoder Block
# ==================================================

class EncoderBlock(nn.Module):
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model)

    def forward(self, x):
        h = self.ln1(x)

        # Q、K、V 都来自 Encoder
        attn_out, _ = self.self_attn(h, h)

        x = x + attn_out
        x = x + self.ffn(self.ln2(x))

        return x


# ==================================================
# 4. Decoder Block
# ==================================================

class DecoderBlock(nn.Module):
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, num_heads)

        self.ln2 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)

        self.ln3 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model)

    def forward(self, x, encoder_output):
        # ① Decoder Masked Self-Attention
        h = self.ln1(x)
        self_out, self_weights = self.self_attn(
            h, h, causal=True
        )
        x = x + self_out

        # ② Cross-Attention
        # Q 来自 Decoder，K、V 来自 Encoder
        cross_out, cross_weights = self.cross_attn(
            self.ln2(x),
            encoder_output,
            causal=False
        )
        x = x + cross_out

        # ③ FFN
        x = x + self.ffn(self.ln3(x))

        return x, self_weights, cross_weights


# ==================================================
# 5. 固定正弦位置编码
# ==================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model=512, max_len=128):
        super().__init__()
        assert d_model % 2 == 0

        pos = torch.arange(
            max_len, dtype=torch.float32
        ).unsqueeze(1)

        div = torch.exp(
            torch.arange(
                0, d_model, 2, dtype=torch.float32
            ) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


# ==================================================
# 6. 完整 Encoder-Decoder Transformer
# ==================================================

class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=512,
        num_heads=8,
        num_layers=2
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position = PositionalEncoding(d_model)

        self.encoder = nn.ModuleList([
            EncoderBlock(d_model, num_heads)
            for _ in range(num_layers)
        ])

        self.decoder = nn.ModuleList([
            DecoderBlock(d_model, num_heads)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, src_ids, tgt_ids):
        # src_ids: (B, Ts)
        # tgt_ids: (B, Tt)

        # Encoder
        enc = self.position(self.embedding(src_ids))
        for block in self.encoder:
            enc = block(enc)

        # Decoder
        dec = self.position(self.embedding(tgt_ids))
        for block in self.decoder:
            dec, self_weights, cross_weights = block(
                dec, enc
            )

        logits = self.lm_head(self.final_norm(dec))

        return logits, self_weights, cross_weights


# ==================================================
# 7. 用两个固定句子验证
# ==================================================

sentences = [
    ["The", "young", "engineer", "writes", "code",
     "and", "solves", "difficult", "problems", "daily"],

    ["The", "experienced", "developer", "reviews", "code",
     "and", "improves", "the", "system", "performance"]
]

vocab = {
    word: i
    for i, word in enumerate(
        sorted(set(w for s in sentences for w in s))
    )
}

# 添加训练用的特殊 Token
vocab["<BOS>"] = len(vocab)
vocab["<EOS>"] = len(vocab)

def encode(tokens):
    return [vocab[w] for w in tokens]

# 两条演示样本：
# Source 句子1 -> Target 句子2
# Source 句子2 -> Target 句子1
# 仅测试计算流程，不是合理的翻译语料
sources = [sentences[0], sentences[1]]
targets = [sentences[1], sentences[0]]

src_ids = torch.tensor([encode(s) for s in sources])

# Teacher Forcing：输入与目标右移对齐
tgt_in = torch.tensor([
    [vocab["<BOS>"]] + encode(t)
    for t in targets
])

tgt_out = torch.tensor([
    encode(t) + [vocab["<EOS>"]]
    for t in targets
])

model = Transformer(
    vocab_size=len(vocab),
    d_model=512,
    num_heads=8,
    num_layers=2
)

# ==================================================
# 8. 一次完整训练步骤
# ==================================================

optimizer = torch.optim.AdamW(
    model.parameters(), lr=3e-4
)

optimizer.zero_grad()

logits, self_weights, cross_weights = model(
    src_ids, tgt_in
)

loss = F.cross_entropy(
    logits.reshape(-1, len(vocab)),
    tgt_out.reshape(-1)
)

loss.backward()
optimizer.step()

print("Encoder input:", src_ids.shape)
print("Decoder input:", tgt_in.shape)
print("Logits:", logits.shape)
print("Decoder self-attention:", self_weights.shape)
print("Cross-attention:", cross_weights.shape)
print("Loss:", loss.item())

# Decoder 必须不能关注未来
Tt = tgt_in.size(1)
future = torch.triu(
    torch.ones(Tt, Tt, dtype=torch.bool),
    diagonal=1
)

assert torch.all(
    self_weights.masked_select(
        future.unsqueeze(0).unsqueeze(0).expand_as(
            self_weights
        )
    ) == 0
)

# Cross-Attention 的参数确实收到梯度
assert model.decoder[0].cross_attn.wq.weight.grad is not None

print("Causal mask verified.")
print("Cross-attention gradient verified.")
print("One training step completed.")

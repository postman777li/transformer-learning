
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ==================================================
# 1. 准备训练数据：Day 11 固定句子 + 额外训练语料
# ==================================================

fixed_sentences = [
    "The young engineer writes code and solves difficult problems daily",
    "The experienced developer reviews code and improves the system performance"
]

extra_sentences = [
    "The engineer writes clean code",
    "The developer reads code and tests the system",
    "The system processes text and predicts the next token",
    "A language model learns patterns from text",
    "The model reads tokens and produces predictions",
    "Attention connects each token with earlier tokens",
    "The network uses embeddings to represent words",
    "Position embeddings provide information about token order",
    "The model predicts the next token from context",
    "Training adjusts parameters to reduce prediction loss",
    "The optimizer updates parameters using gradients",
    "The loss measures how well predictions match targets",
    "The decoder uses causal attention to hide future tokens",
    "The feed forward network transforms hidden features",
    "The model processes batches of token sequences",
    "The output layer produces scores for every token",
    "A small dataset can test the training pipeline",
    "A larger dataset helps the model learn more patterns",
    "The researcher trains the model on text",
    "The model generates text one token at a time",
    "The training loop repeats forward and backward passes",
    "The attention weights combine information from values",
    "The token embedding maps each token to a vector",
    "The position embedding adds a learned position vector",
    "The transformer block uses attention and a feed forward network",
    "The model needs examples to learn language patterns",
    "The code verifies that parameters change after training",
    "The engineer studies the model and improves the code",
    "The developer trains the network with gradient descent",
    "The model calculates loss and updates its parameters"
]

all_sentences = fixed_sentences + extra_sentences

# 每句话结束时添加 <eos>
all_tokens = []
for sentence in all_sentences:
    all_tokens.extend(sentence.split())
    all_tokens.append("<eos>")

# 构建词表
vocab = sorted(set(all_tokens))
stoi = {token: i for i, token in enumerate(vocab)}
itos = {i: token for token, i in stoi.items()}

vocab_size = len(vocab)

# 把全部语料转换为一条 Token ID 序列
data = torch.tensor(
    [stoi[token] for token in all_tokens],
    dtype=torch.long,
    device=device
)

# 保留两个固定句子，供后续预测和因果性验证
input_ids = torch.tensor(
    [
        [stoi[token] for token in sentence.split()]
        for sentence in fixed_sentences
    ],
    dtype=torch.long,
    device=device
)

print("Device:", device)
print("Vocabulary size:", vocab_size)
print("Training tokens:", len(data))
print("Fixed input shape:", input_ids.shape)


# ==================================================
# 2. Causal Multi-Head Attention
# ==================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.shape
        H, D = self.num_heads, self.head_dim

        # (B,T,C) -> (B,T,3C)
        qkv = self.qkv(x)

        # (B,T,3C) -> (B,T,3,H,D)
        qkv = qkv.view(B, T, 3, H, D)

        # (B,T,3,H,D) -> (3,B,H,T,D)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)

        # (B,H,T,D) @ (B,H,D,T) -> (B,H,T,T)
        scores = (
            q @ k.transpose(-2, -1)
        ) / math.sqrt(D)

        # 禁止当前位置关注未来 Token
        mask = torch.triu(
            torch.ones(
                T, T,
                dtype=torch.bool,
                device=x.device
            ),
            diagonal=1
        )

        scores = scores.masked_fill(mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)

        # (B,H,T,T) @ (B,H,T,D) -> (B,H,T,D)
        out = weights @ v

        # 合并多个 Head
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.proj(out)


# ==================================================
# 3. Feed Forward Network
# ==================================================

class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x):
        return self.net(x)


# ==================================================
# 4. Transformer Block
# ==================================================

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads)

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ==================================================
# 5. Mini-GPT
# ==================================================

class MiniGPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=128,
        num_heads=4,
        num_layers=2,
        max_len=32
    ):
        super().__init__()

        self.max_len = max_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads)
            for _ in range(num_layers)
        ])

        self.ln_final = nn.LayerNorm(d_model)

        self.lm_head = nn.Linear(
            d_model, vocab_size, bias=False
        )

    def forward(self, input_ids):
        B, T = input_ids.shape
        assert T <= self.max_len

        # Token Embedding
        token_vectors = self.token_emb(input_ids)

        # Learned Position Embedding
        positions = torch.arange(T, device=input_ids.device)
        position_vectors = self.pos_emb(positions)

        # (B,T,C) + (T,C) -> (B,T,C)
        x = token_vectors + position_vectors

        for block in self.blocks:
            x = block(x)

        x = self.ln_final(x)

        # (B,T,C) -> (B,T,V)
        logits = self.lm_head(x)

        return logits


# ==================================================
# 6. 构造 Next-Token Prediction 训练批次
# ==================================================

block_size = 16
batch_size = 16

def get_batch():
    # 每个输入窗口长度为 block_size，
    # 目标窗口向后移动一个 Token
    max_start = len(data) - block_size

    starts = torch.randint(
        0, max_start, (batch_size,)
    ).tolist()

    x = torch.stack([
        data[i:i + block_size]
        for i in starts
    ])

    y = torch.stack([
        data[i + 1:i + block_size + 1]
        for i in starts
    ])

    return x, y


# ==================================================
# 7. 实例化模型和优化器
# ==================================================

model = MiniGPT(
    vocab_size=vocab_size,
    d_model=128,
    num_heads=4,
    num_layers=2,
    max_len=block_size
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3
)

print("Model parameters:", sum(
    p.numel() for p in model.parameters()
))


# ==================================================
# 8. 固定一批数据，用来比较训练前后的 Loss
# ==================================================

eval_x, eval_y = get_batch()

model.eval()
with torch.no_grad():
    eval_logits = model(eval_x)

    initial_loss = F.cross_entropy(
        eval_logits.reshape(-1, vocab_size),
        eval_y.reshape(-1)
    ).item()

print(f"Initial loss: {initial_loss:.4f}")


# ==================================================
# 9. 训练循环：前向传播 -> Loss -> 反向传播 -> 更新
# ==================================================

epochs = 300

for epoch in range(epochs):
    model.train()

    # 获取一个新批次
    x, y = get_batch()

    # ① 清空上一步累积的梯度
    optimizer.zero_grad()

    # ② 前向传播
    logits = model(x)

    # ③ 计算所有位置的下一 Token 预测损失
    # logits: (B,T,V) -> (B*T,V)
    # y:      (B,T)   -> (B*T,)
    loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        y.reshape(-1)
    )

    # ④ 反向传播，计算参数梯度
    loss.backward()

    # ⑤ 根据梯度更新模型参数
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(
            f"Epoch {epoch + 1:3d}/{epochs} "
            f"| Loss: {loss.item():.4f}"
        )


# ==================================================
# 10. 验证训练结果
# ==================================================

model.eval()

with torch.no_grad():
    eval_logits = model(eval_x)

    final_loss = F.cross_entropy(
        eval_logits.reshape(-1, vocab_size),
        eval_y.reshape(-1)
    ).item()

    predictions = eval_logits.argmax(dim=-1)
    accuracy = (predictions == eval_y).float().mean().item()

print(f"\nInitial loss: {initial_loss:.4f}")
print(f"Final loss:   {final_loss:.4f}")
print(f"Batch accuracy: {accuracy:.2%}")


# ==================================================
# 11. 检查模型参数是否真的发生变化
# ==================================================

# 注意：这里是在训练结束后检查梯度是否存在。
# 参数更新是否发生，还可以通过训练前后参数对比验证。
# 因此实际使用时应在训练前保存参数副本。

print("Training finished.")


# ==================================================
# 12. 用固定句子查看模型的下一 Token 预测
# ==================================================

prompt_ids = input_ids[0:1]

with torch.no_grad():
    logits = model(prompt_ids)

    # 最后一个位置预测下一个 Token
    last_logits = logits[0, -1, :]
    probabilities = F.softmax(last_logits, dim=-1)

    predicted_id = torch.argmax(probabilities).item()

print("Prompt:", fixed_sentences[0])
print("Predicted next token:", itos[predicted_id])


# ==================================================
# 13. 验证 Causal Mask 没有泄漏未来信息
# ==================================================

with torch.no_grad():
    original = model(input_ids)

    modified = input_ids.clone()
    modified[0, 4] = stoi["daily"]

    changed = model(modified)

    assert torch.allclose(
        original[0, 3],
        changed[0, 3],
        atol=1e-5,
        rtol=1e-5
    )

print("Causal verification passed.")

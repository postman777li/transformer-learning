
import os

# 限制数值计算线程，减少部分 Windows 环境的资源分配问题
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
torch.set_num_threads(1)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ==================================================
# 1. 准备训练数据
# ==================================================

fixed_sentences = [
    "The young engineer writes code and solves difficult problems daily",
    "The experienced developer reviews code and improves the system performance"
]

extra_sentences = [
    "The engineer writes clean code",
    "The developer reads code and tests the system",
    "The model predicts the next token",
    "Training adjusts parameters to reduce loss",
    "Attention combines information from earlier tokens",
    "The optimizer updates parameters using gradients",
    "The model learns patterns from text",
    "The network uses embeddings to represent words",
    "The decoder uses causal attention",
    "The output layer predicts the next token",
    "The researcher trains the model on text",
    "The model generates text one token at a time",
    "The training loop repeats forward and backward passes",
    "The transformer uses attention and a feed forward network",
    "The model processes batches of token sequences"
]

all_sentences = fixed_sentences + extra_sentences

all_tokens = []

for sentence in all_sentences:
    all_tokens.extend(sentence.split())
    all_tokens.append("<eos>")

vocab = sorted(set(all_tokens))
stoi = {token: i for i, token in enumerate(vocab)}
itos = {i: token for token, i in stoi.items()}

vocab_size = len(vocab)

train_data = torch.tensor(
    [stoi[token] for token in all_tokens],
    dtype=torch.long,
    device=device
)

print("Device:", device)
print("Vocabulary size:", vocab_size)
print("Training tokens:", len(train_data))


# ==================================================
# 2. Causal Self-Attention + KV Cache
# ==================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        x,
        past_key_value=None,
        use_cache=False
    ):
        B, T, C = x.shape
        H, D = self.num_heads, self.head_dim

        # ① 当前输入生成 Q、K、V
        # (B,T,C) -> (B,T,3C)
        qkv = self.qkv(x)

        # (B,T,3C) -> (B,T,3,H,D)
        qkv = qkv.view(B, T, 3, H, D)

        # (B,T,3,H,D) -> (3,B,H,T,D)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        q, k, v = qkv.unbind(dim=0)

        # ② 如果有历史缓存，拼接旧 K、V
        if past_key_value is not None:
            if T != 1:
                raise ValueError(
                    "本示例的 KV Cache 解码每次输入一个新 Token"
                )

            past_k, past_v = past_key_value

            # 只沿序列长度维度追加新 Token
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)

        # 此时：
        # q: (B,H,T,D)
        # k: (B,H,历史长度+T,D)
        # v: (B,H,历史长度+T,D)

        # ③ 计算 Attention Score
        scores = (
            q @ k.transpose(-2, -1)
        ) / math.sqrt(D)

        # ④ 首次处理完整前缀时，使用 Causal Mask
        if past_key_value is None:
            mask = torch.triu(
                torch.ones(
                    T, T,
                    dtype=torch.bool,
                    device=x.device
                ),
                diagonal=1
            )

            scores = scores.masked_fill(
                mask, float("-inf")
            )

        # 增量生成时，K、V 只有历史位置和当前位置，
        # 因此不需要再屏蔽未来位置。

        weights = F.softmax(scores, dim=-1)

        # ⑤ Attention 加权求和
        out = weights @ v

        # (B,H,T,D) -> (B,T,C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        out = self.proj(out)

        # ⑥ 返回输出和更新后的缓存
        if use_cache:
            return out, (k, v)

        return out


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
# 4. Transformer Block：逐层传递缓存
# ==================================================

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads)

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model)

    def forward(
        self,
        x,
        past_key_value=None,
        use_cache=False
    ):
        if use_cache:
            attn_out, present_kv = self.attn(
                self.ln1(x),
                past_key_value=past_key_value,
                use_cache=True
            )
        else:
            attn_out = self.attn(self.ln1(x))

        # Attention 残差连接
        x = x + attn_out

        # FFN 残差连接
        x = x + self.ffn(self.ln2(x))

        if use_cache:
            return x, present_kv

        return x


# ==================================================
# 5. Mini-GPT：支持 KV Cache
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

    def forward(
        self,
        input_ids,
        past_key_values=None,
        use_cache=False
    ):
        B, T = input_ids.shape

        # ① 确定历史缓存长度
        if past_key_values is None:
            past_key_values = [None] * len(self.blocks)
            past_len = 0
        else:
            past_len = past_key_values[0][0].size(-2)

        assert past_len + T <= self.max_len

        # ② 位置编号必须接着历史长度往后走
        positions = torch.arange(
            past_len,
            past_len + T,
            device=input_ids.device
        )

        x = (
            self.token_emb(input_ids)
            + self.pos_emb(positions)
        )

        # ③ 每一层维护自己的 K、V 缓存
        new_key_values = []

        for block, past_kv in zip(
            self.blocks, past_key_values
        ):
            if use_cache:
                x, present_kv = block(
                    x,
                    past_key_value=past_kv,
                    use_cache=True
                )

                new_key_values.append(present_kv)
            else:
                x = block(x)

        # ④ 输出词表 logits
        x = self.ln_final(x)
        logits = self.lm_head(x)

        if use_cache:
            return logits, new_key_values

        return logits


# ==================================================
# 6. 训练 Mini-GPT
# ==================================================

block_size = 16
batch_size = 8

model = MiniGPT(
    vocab_size=vocab_size,
    d_model=128,
    num_heads=4,
    num_layers=2,
    max_len=32
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3
)


def get_batch():
    starts = torch.randint(
        0,
        len(train_data) - block_size,
        (batch_size,)
    ).tolist()

    x = torch.stack([
        train_data[i:i + block_size]
        for i in starts
    ])

    y = torch.stack([
        train_data[i + 1:i + block_size + 1]
        for i in starts
    ])

    return x, y


before = model.token_emb.weight.detach().clone()

model.train()

for step in range(301):
    x, y = get_batch()

    optimizer.zero_grad()

    logits = model(x)

    loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        y.reshape(-1)
    )

    loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(
            f"Step {step:3d} | Loss: {loss.item():.4f}"
        )

after = model.token_emb.weight.detach()

print(
    "Parameters updated:",
    not torch.equal(before, after)
)


# ==================================================
# 7. Day 13：普通生成，不使用 KV Cache
# ==================================================

@torch.no_grad()
def generate_without_cache(
    model,
    input_ids,
    max_new_tokens=10
):
    model.eval()

    generated = input_ids

    for _ in range(max_new_tokens):
        context = generated[:, -model.max_len:]

        logits = model(context)
        next_logits = logits[:, -1, :]

        next_token = next_logits.argmax(
            dim=-1,
            keepdim=True
        )

        generated = torch.cat(
            [generated, next_token],
            dim=1
        )

        if (next_token == stoi["<eos>"]).all():
            break

    return generated


# ==================================================
# 8. Day 14：使用 KV Cache 生成
# ==================================================

@torch.no_grad()
def generate_with_cache(
    model,
    input_ids,
    max_new_tokens=10,
    temperature=1.0,
    top_k=5,
    top_p=0.9,
    do_sample=True
):
    model.eval()

    if temperature <= 0:
        raise ValueError("temperature 必须大于 0")

    generated = input_ids

    # 首次处理完整提示词，建立各层 K、V 缓存
    logits, past_key_values = model(
        input_ids,
        use_cache=True
    )

    for step in range(max_new_tokens):
        # 只使用最后一个位置预测下一个 Token
        next_logits = logits[:, -1, :] / temperature

        # Top-k
        if top_k is not None:
            k = min(top_k, next_logits.size(-1))

            threshold = torch.topk(
                next_logits, k, dim=-1
            ).values[:, -1:]

            next_logits = next_logits.masked_fill(
                next_logits < threshold,
                float("-inf")
            )

        # Top-p
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(
                next_logits,
                descending=True,
                dim=-1
            )

            sorted_probs = F.softmax(
                sorted_logits, dim=-1
            )

            cumulative_probs = sorted_probs.cumsum(dim=-1)

            remove = cumulative_probs > top_p

            # 保留第一个使累计概率超过 top_p 的 Token
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False

            indices_to_remove = torch.zeros_like(
                next_logits,
                dtype=torch.bool
            )

            indices_to_remove.scatter_(
                dim=-1,
                index=sorted_indices,
                src=remove
            )

            next_logits = next_logits.masked_fill(
                indices_to_remove,
                float("-inf")
            )

        # 选择下一个 Token
        if do_sample:
            probs = F.softmax(next_logits, dim=-1)

            next_token = torch.multinomial(
                probs,
                num_samples=1
            )
        else:
            next_token = next_logits.argmax(
                dim=-1,
                keepdim=True
            )

        generated = torch.cat(
            [generated, next_token],
            dim=1
        )

        if (next_token == stoi["<eos>"]).all():
            break

        if step == max_new_tokens - 1:
            break

        # 只把刚生成的 Token 输入模型。
        # 历史 Token 的 K、V 已经保存在 past_key_values 中。
        logits, past_key_values = model(
            next_token,
            past_key_values=past_key_values,
            use_cache=True
        )

    return generated


# ==================================================
# 9. 验证 KV Cache 计算正确
# ==================================================

model.eval()

prompt_tokens = [
    "The", "young", "engineer", "writes"
]

prompt_ids = torch.tensor(
    [[stoi[token] for token in prompt_tokens]],
    dtype=torch.long,
    device=device
)

with torch.no_grad():
    # 首次处理完整提示词
    full_logits = model(prompt_ids)

    cached_logits, past_key_values = model(
        prompt_ids,
        use_cache=True
    )

    # 两种方式最后一个位置的 logits 应该一致
    assert torch.allclose(
        full_logits[:, -1, :],
        cached_logits[:, -1, :],
        atol=1e-5,
        rtol=1e-5
    )

    # 模拟下一个 Token 已经确定为 code
    next_token = torch.tensor(
        [[stoi["code"]]],
        dtype=torch.long,
        device=device
    )

    # KV Cache：只输入新 Token
    incremental_logits, _ = model(
        next_token,
        past_key_values=past_key_values,
        use_cache=True
    )

    # 普通方式：重新计算完整序列
    full_sequence = torch.cat(
        [prompt_ids, next_token],
        dim=1
    )

    recomputed_logits = model(full_sequence)

    # 两种方式在新 Token 位置的 logits 应一致
    assert torch.allclose(
        incremental_logits[:, -1, :],
        recomputed_logits[:, -1, :],
        atol=1e-5,
        rtol=1e-5
    )

print("KV Cache verification passed.")


# ==================================================
# 10. 实际生成文本
# ==================================================

result = generate_with_cache(
    model,
    prompt_ids,
    max_new_tokens=10,
    temperature=0.8,
    top_k=5,
    top_p=0.9,
    do_sample=True
)


def decode(token_ids):
    words = []

    for token_id in token_ids:
        token = itos[int(token_id)]

        if token == "<eos>":
            break

        words.append(token)

    return " ".join(words)


print("\nPrompt:", decode(prompt_ids[0].tolist()))
print("Generated:", decode(result[0].tolist()))

new_tokens = result[0, prompt_ids.size(1):]

print("New tokens:", decode(new_tokens.tolist()))


# ==================================================
# 11. 保存训练后的模型
# ==================================================

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "stoi": stoi,
        "itos": itos,
        "config": {
            "vocab_size": vocab_size,
            "d_model": 128,
            "num_heads": 4,
            "num_layers": 2,
            "max_len": 32
        }
    },
    "day14_minigpt.pt"
)

print("\nModel saved to day14_minigpt.pt")

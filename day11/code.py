import torch
import torch.nn as nn
import math
import torch.nn.functional as F
#两句话 每句话10个token 每个token512维 B=2 T=10 C=512 H=10 10个头
sentences = [
    ["The", "young", "engineer", "writes", "code",
     "and", "solves", "difficult", "problems", "daily"],

    ["The", "experienced", "developer", "reviews", "code",
     "and", "improves", "the", "system", "performance"]
]

#1.用token建立词表，并且给每个token分配ID
vocab=[]
for sentence in sentences:
    for s in sentence:
        if s not in vocab:
            vocab.append(s)
vocab.sort()

stoi={}
for i in range(len(vocab)):
    stoi[vocab[i]]=i

#2.把上面的sentences改成编号的形式
input_ids_list=[]
for sentence in sentences:
    sentence_ids=[]
    for token in sentence:
        token_id=stoi[token]
        sentence_ids.append(token_id)
    input_ids_list.append(sentence_ids)
input_ids=torch.tensor(input_ids_list, dtype=torch.long)

# 3. Causal Multi-Head Attention
class CausalSelfAttention(nn.Module):
    #__init__ 初始化的是网络结构和需要训练的参数
    def __init__(self,d_model,num_heads):
        super().__init__()
        #nn.Linear(in_features, out_features, bias=True)
        #y=wX+b 
        #不用self那就只是init的局部变量了 forward用不了
        self.num_heads=num_heads
        self.head_dim=d_model//num_heads

        self.qkv=nn.Linear(d_model,3*d_model)
        self.proj=nn.Linear(d_model,d_model)
    def forward(self,x):
        #x.shape = (2, 10, 512)
        B,T,C=x.shape
        H,D=self.num_heads,self.head_dim

        #(2,10,512)->(2,10,1536)  (B,T,C)->(B,T,3C)
        qkv=self.qkv(x)
        # -> (B,T,3,H,D)
        qkv=qkv.view(B,T,3,H,D)
        # -> (3,B,H,T,D)
        qkv=qkv.permute(2, 0, 3, 1, 4)

        # q.shape = (2, 8, 10, 64)
        # k.shape = (2, 8, 10, 64)
        # v.shape = (2, 8, 10, 64)
        q,k,v=qkv.unbind(dim=0)

        #scores.shape  # (B, H, T, T)
        scores=(q@k.transpose(-2,-1))/math.sqrt(D)

        # True 表示未来位置（禁止关注）
        mask=torch.triu(
            torch.ones(T,T,dtype=torch.bool,device=x.device),
            diagonal=1
        )
        #masked_fill(mask, value)：在 mask 为 True 的位置替换数值。
        scores = scores.masked_fill(mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)

        # (B,H,T,T) @ (B,H,T,D)
        # -> (B,H,T,D)
        out = weights @ v

        # 拼接 Heads -> (B,T,C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.proj(out)

# ==========================
# 3. FFN
# ==========================

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


# ==========================
# 4. Transformer Block
# ==========================

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


# ==========================
# 5. Mini-GPT
# ==========================

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
        self.token_emb = nn.Embedding(vocab_size, d_model)

        # 可学习位置 Embedding
        self.pos_emb = nn.Embedding(max_len, d_model)

        # 多个 Transformer Block
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads)
            for _ in range(num_layers)
        ])

        self.ln_final = nn.LayerNorm(d_model)

        # LM Head: 隐藏向量 -> 词表 Logits
        self.lm_head = nn.Linear(
            d_model, vocab_size, bias=False
        )

    def forward(self, input_ids):
        B, T = input_ids.shape
        assert T <= self.max_len

        # ① Token Embedding
        token_vectors = self.token_emb(input_ids)

        # ② Position Embedding
        positions = torch.arange(T, device=input_ids.device)
        position_vectors = self.pos_emb(positions)

        # (B,T,C) + (T,C) -> (B,T,C)
        x = token_vectors + position_vectors

        # ③ Transformer Blocks
        for block in self.blocks:
            x = block(x)

        # ④ Final LayerNorm
        x = self.ln_final(x)

        # ⑤ LM Head
        logits = self.lm_head(x)

        return logits

# ==========================
# 6. 实例化与验证
# ==========================
model=MiniGPT(
    vocab_size=len(vocab),
    d_model=512,
    num_heads=8,
    num_layers=4,
    max_len=128
)

logits=model(input_ids)

print("Token IDs:", input_ids.shape)
print("Logits:", logits.shape)
print("Parameters:", sum(p.numel() for p in model.parameters()))

#取第一个句子的最后一个位置的预测分数
last_logits=logits[0,-1,:]
probs=F.softmax(last_logits,dim=-1)

predicted_id=torch.argmax(probs).item()
print("Predicted next token:", vocab[predicted_id])

# 验证因果性：
# 更改第 5 个 Token，不应影响第 4 个位置的输出

model.eval()
with torch.no_grad():
    original = model(input_ids)

    modified = input_ids.clone()
    modified[0, 4] = stoi["daily"]

    changed = model(modified)

    assert torch.allclose(
        original[0, 3], changed[0, 3],
        atol=1e-5, rtol=1e-5
    )

print("Causal verification passed.")

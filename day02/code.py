import torch
import torch.nn as nn

#两句话 一句话三个token 这是token的id
tokens_id=torch.tensor([
    [12, 35, 208],
    [51, 9, 300]
])

#词表里1000个词 每个词是4维
embedding=nn.Embedding(
    num_embeddings=1000,
    embedding_dim=4
)

x=embedding(tokens_id)

print(tokens_id.shape)
print(x.shape)
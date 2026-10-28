import torch
import torch.nn.functional as F

tokens=torch.tensor([
    [1.0, 0.0],   # A
    [0.0, 1.0],   # B
    [1.0, 1.0]    # C
])

A=tokens[0]
# 2. A 和所有 token 计算相似度
scores=tokens @ A
# 3. 转换成 attention 权重
weights=F.softmax(scores,dim=0)
# 4. 根据权重聚合所有 token 的信息
output=weights@tokens

print("tokens:")
print(tokens)

print("\nA:")
print(A)

print("\nscores:")
print(scores)

print("\nattention weights:")
print(weights)

print("\nattention weights sum:")
print(weights.sum())

print("\nattention output:")
print(output)
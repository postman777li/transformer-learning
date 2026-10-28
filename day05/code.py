
import torch
import math

# 1. 招聘需求 Q
Q = torch.tensor([2.0, 1.0])

# 2. 两位候选人的 K
K_ming = torch.tensor([9.0, 1.0])
K_hong = torch.tensor([3.0, 4.5])

# 3. 两位候选人的 V
V_ming = torch.tensor([9.0, 2.0])
V_hong = torch.tensor([3.0, 9.0])

# 4. 把 K、V 组合成矩阵
K = torch.stack([K_ming, K_hong])
V = torch.stack([V_ming, V_hong])

print("Q =", Q)
print("K =\n", K)
print("V =\n", V)

# 5. 一次性计算所有匹配分数
scores = Q @ K.T

print("\n原始匹配分数：", scores)

# 6. Scaled Dot-Product
d_k = K.shape[-1]
scaled_scores = scores / math.sqrt(d_k)

print("缩放后的分数：", scaled_scores)

# 7. Softmax 得到权重
weights = torch.softmax(scaled_scores, dim=-1)

print("注意力权重：", weights)

# 8. 加权汇总 V
output = weights @ V

print("最终输出：", output)
print("输出 shape：", output.shape)

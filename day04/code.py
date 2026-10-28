import torch

#这是这三个词的embedding向量
animal=torch.tensor([1.0,0.0])
street=torch.tensor([0.0,1.0])
it=torch.tensor([0.8,0.2])

print(animal)
print(street)
print(it)

#就是第一维不变 第二维变成2倍 目的就是在这个例子中更加关注第二维 
Wq=torch.tensor([
    [1.0, 0.0],
    [0.0, 2.0]
])

Q_it=it@Wq
print("it的query:",Q_it)

Wk=torch.tensor([
    [2.0, 0.0],
    [0.0, 1.0]
])

K_animal=animal@Wk
K_street=street@Wk

print("animal 的 K =", K_animal)
print("street 的 K =", K_street)

score_animal=torch.dot(Q_it,K_animal)
score_street=torch.dot(Q_it,K_street)

print("it 对 animal 的匹配分数 =", score_animal.item())
print("it 对 street 的匹配分数 =", score_street.item())

Wv = torch.tensor([
    [1.0, 1.0],
    [0.0, 1.0]
])

V_animal = animal @ Wv
V_street = street @ Wv

print("animal 的 V =", V_animal)
print("street 的 V =", V_street)

#把刚才算的两个Q与K的分数放在一个向量[1.6,0.4]
scores = torch.stack([score_animal, score_street])
weights=torch.softmax(scores,dim=0)

output_it = (
    weights[0] * V_animal
    + weights[1] * V_street
)

print("注意力权重 =", weights)
print("it 融合后的信息 =", output_it)
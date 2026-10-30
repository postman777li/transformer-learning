# Day 3：Attention 核心直觉

> 核心过程：计算匹配分数 → 转换成权重 → 加权汇总信息。

## 1. 从代词理解 Attention

> The animal didn't cross the street because **it** was too tired.
>
> 那只动物没有过马路，因为它太累了。

理解 `it` 时，需要联系 `animal` 和 `tired` 等上下文。Attention 的思路是：**让当前位置根据匹配程度，分配关注权重，再汇总相关位置的信息。**

例如假设 `it` 对四个位置的权重如下：

| Token | animal | street | it | tired |
|---|---:|---:|---:|---:|
| 关注权重 | 60% | 5% | 10% | 25% |

这些数值只是直觉示意，并非实际计算结果，也不能把注意力权重直接当成代词指代的证明。

## 2. 用三个向量手算一次 Attention

设三个 Token 的向量为：

```text
A = [1, 0]
B = [0, 1]
C = [1, 1]
```

目标：让 A 参考 A、B、C，得到新表示。

### 第一步：点积计算匹配分数

```text
A · A = 1×1 + 0×0 = 1
A · B = 1×0 + 0×1 = 0
A · C = 1×1 + 0×1 = 1

scores = [1, 0, 1]
```

点积衡量当前向量表示下的匹配程度，也受向量长度影响，不等同于真实语义相似度。

### 第二步：Softmax 转换为权重

$$
w_i=\frac{e^{s_i}}{\sum_j e^{s_j}}
$$

```text
softmax([1, 0, 1]) ≈ [0.4223, 0.1554, 0.4223]
```

**Softmax 将这些有限分数转换为正权重，权重之和为 1；分数越大，权重越大。** 显示值经过舍入，求和可能略有误差。

### 第三步：加权汇总向量

$$
A_{\mathrm{new}}=w_AA+w_BB+w_CC
$$

```text
A_new ≈ 0.4223×[1,0] + 0.1554×[0,1] + 0.4223×[1,1]
      ≈ [0.8446, 0.5777]
```

原来的 A 是 `[1,0]`，新向量融合了三个位置的信息。**Attention 并非只选中分数最高的 Token，而是按一组权重汇总信息。**

## 3. 对应到专业概念

| 概念 | 对应含义 |
|---|---|
| Attention | 根据匹配分数产生权重，对信息进行加权汇总 |
| Contextual Representation（上下文化表示） | 融合上下文后得到、随语境变化的表示 |
| Self-Attention（自注意力） | 查询和被汇总的信息来自同一序列 |

上下文化表示可以用 `bank` 理解：

- `I deposited money at the bank.`：结合 `money` 等上下文，表示银行。
- `I sat on the river bank.`：结合 `river` 等上下文，表示河岸。

如果两处 `bank` 对应同一个 Token ID，其初始 Token Embedding 相同；经过上下文处理后，表示可以不同。

前面的手算只更新了 A。对全部位置做 Self-Attention，还要分别计算 B、C 的权重和新表示。无 Mask 时可以相互关注；GPT 的因果 Mask 则只允许关注自己及之前的位置。

### 与后续课程的衔接

本例直接用原始向量匹配、汇总，没有学习到的 Q/K/V 投影，也没有缩放。标准形式是：

$$
\mathrm{Attention}(Q,K,V)
=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

Q 用于查询，K 用于匹配，V 提供被汇总的信息。具体投影与缩放留到 Day 4、Day 5；本日重点是理解“分数 → 权重 → 汇总”。

## 4. 对应代码与 Shape

[code.py](code.py) 保留原始练习，核心三行是：

```python
scores = tokens @ A                 # [3,2] @ [2] → [3]
weights = F.softmax(scores, dim=0)   # [3] → [3]
output = weights @ tokens           # [3] @ [3,2] → [2]
```

第一步得到 A 对三个位置的分数；最后一步将三个二维向量汇总成一个二维向量。

从仓库根目录运行（需安装 PyTorch）：

```bash
python day03/code.py
```

根据代码推导，关键输出为：

```text
scores:  [1.0, 0.0, 1.0]
weights: [0.4223, 0.1554, 0.4223]
output:  [0.8446, 0.5777]
```

以上为数学推导值，尚未在此环境执行脚本验证。

## 5. 自测

1. 把 `A = tokens[0]` 换成 `tokens[1]` 或 `tokens[2]`，分别计算 B、C 的新表示。
2. 解释为什么 `[3,2] @ [2]` 得到三个分数，而 `[3] @ [3,2]` 得到二维向量。
3. 进阶：用 `tokens @ tokens.T` 一次得到 `[3,3]` 的分数矩阵，沿最后一维做 Softmax，再汇总全部位置的信息。

**记住：Attention 用权重汇总上下文，让 Token 得到新的表示。**

---

[返回学习导航](../README.md)

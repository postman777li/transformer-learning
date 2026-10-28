# Transformer：16 天学习记录

按天整理 Transformer 学习过程：从 Attention 的直觉与手算，到 Mini-GPT、训练、生成及现代 LLM。

每天保存两个文件：

- `notes.md`：概念、帮助理解的例子、必要的公式和个人总结。
- `code.py`：当天的代码实践。当前全部留空，随后逐日补充。

笔记采用“具体例子 → 数字推导 → 专业概念”的方式，保留便于理解的类比，并说明类比的适用边界。

## 当前整理进度

目前已上传 Day 6 的多头直觉、招聘例子、手算过程与核心公式；Day 6 后续内容及其他天数预留待补。下表记录资料整理进度，不代表实际学习进度。

## 16 天导航

| 天数 | 主题 | 当天要掌握的内容 | 计划代码 | 文件入口 | 笔记状态 |
|---|---|---|---|---|---|
| Day 1 | Transformer 为什么存在 | RNN/LSTM 的问题、序列建模、Attention 想解决什么 | 简单张量练习 | [笔记](day01/notes.md) · [代码](day01/code.py) | 待整理 |
| Day 2 | Embedding + 必要矩阵基础 | token→向量、矩阵乘法、dot product、shape | 手写 embedding 示例 | [笔记](day02/notes.md) · [代码](day02/code.py) | 待整理 |
| Day 3 | Attention 核心直觉 | 一个 token 怎么关注另一个 token | 手算最简单 Attention | [笔记](day03/notes.md) · [代码](day03/code.py) | 待整理 |
| Day 4 | Q / K / V 彻底搞懂 | QKV 为什么存在、Wq/Wk/Wv 为什么要训练 | 从零写 Self-Attention | [笔记](day04/notes.md) · [代码](day04/code.py) | 待整理 |
| Day 5 | Scaled Dot-Product Attention | QKᵀ/√d、softmax、每个维度怎么变化 | 完整 Attention 函数 | [笔记](day05/notes.md) · [代码](day05/code.py) | 待整理 |
| Day 6 | Multi-Head Attention | 多头到底多在哪里、为什么 concatenate | 手写 MultiHeadAttention | [笔记](day06/notes.md) · [代码](day06/code.py) | 例子与概念已整理，后续待补 |
| Day 7 | Position + Mask | 位置编码、causal mask、为什么 GPT 不会偷看答案 | 手写 mask | [笔记](day07/notes.md) · [代码](day07/code.py) | 待整理 |
| Day 8 | FFN / Residual / LayerNorm | Transformer Block 另外三块的作用 | 手写 TransformerBlock | [笔记](day08/notes.md) · [代码](day08/code.py) | 待整理 |
| Day 9 | 原版 Transformer 全架构 | Encoder、Decoder、Cross Attention | 搭完整 Transformer | [笔记](day09/notes.md) · [代码](day09/code.py) | 待整理 |
| Day 10 | GPT 是怎么从 Transformer 变来的 | Decoder-only、next-token prediction | 搭 MiniGPT 骨架 | [笔记](day10/notes.md) · [代码](day10/code.py) | 待整理 |
| Day 11 | 从零写 Mini-GPT | Embedding→Blocks→LM Head 全链路 | 自己完成 GPT | [笔记](day11/notes.md) · [代码](day11/code.py) | 待整理 |
| Day 12 | Transformer 怎么训练 | logits、cross entropy、teacher forcing、batch | 真正训练 Mini-GPT | [笔记](day12/notes.md) · [代码](day12/code.py) | 待整理 |
| Day 13 | Transformer 怎么生成文字 | autoregressive、temperature、top-k/top-p | 写 generate() | [笔记](day13/notes.md) · [代码](day13/code.py) | 待整理 |
| Day 14 | KV Cache + 推理机制 | 为什么生成慢、KV 到底缓存什么 | 实现/分析 KV Cache | [笔记](day14/notes.md) · [代码](day14/code.py) | 待整理 |
| Day 15 | 现代 LLM：Llama | RoPE、RMSNorm、SwiGLU、GQA 及其改动原因 | 看懂 Llama Block | [笔记](day15/notes.md) · [代码](day15/code.py) | 待整理 |
| Day 16 | 高手课 + 总考试 | FlashAttention、复杂度、长上下文、读源码方法 | 独立手写核心模块 | [笔记](day16/notes.md) · [代码](day16/code.py) | 待整理 |

## 目录结构

```text
transformer-learning/
├── README.md
├── day01/
│   ├── notes.md
│   └── code.py
├── …
└── day16/
    ├── notes.md
    └── code.py
```

## 后续更新方式

1. 把当天概念、例子与理解整理到对应的 `notes.md`。
2. 把实际编写的代码放入同目录的 `code.py`，并在笔记中补充运行方式与结果。
3. 更新上方导航中的笔记状态。

代码尚未整理，当前没有统一的运行命令或依赖清单；待实际代码上传后再补充。

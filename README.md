# Transformer (Attention Is All You Need)
### 选择语言 | Language
[中文简介](#简介) | [English](#introduction)
### 结果 | Result

<img width="839" height="225" alt="image" src="https://github.com/user-attachments/assets/327f09d3-eb2c-4c3f-a5e2-6285ac5fe904" />

<img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/fe048dae-7379-4504-8eb6-a788cc566ffd" />

---
## 简介
Transformer 由 **Ashish Vaswani, Noam Shazeer, Niki Parmar 等人** 于 2017 年提出（NeurIPS 2017），是序列建模领域的颠覆性革新。其核心贡献在于**完全摒弃循环结构，仅基于自注意力机制实现序列到序列建模**，解决了传统 RNN/LSTM 系列模型无法并行计算、长距离依赖传递衰减的核心痛点，重新定义了自然语言处理、语音、计算机视觉等几乎所有序列任务的基础架构。

在 Transformer 出现之前，主流序列模型均依赖循环神经网络按时间步顺序运算，训练无法并行，且长距离语义传递效率低下。即使引入注意力机制作为辅助，循环结构本身仍是性能与速度的双重瓶颈。Transformer 首次将**自注意力**作为核心计算单元，通过缩放点积注意力同时计算序列内所有位置的关联权重，配合多头机制捕捉不同语义子空间的信息，结合位置编码保留序列顺序信息，实现了全并行的序列建模，训练效率与建模能力同时获得量级提升。

在 WMT 2014 英德翻译任务上，基础版 Transformer 取得了 **28.4 BLEU**，超越当时所有 RNN 与卷积模型，且训练成本仅为之前最优模型的 1/4。本复现基于 PyTorch 从零实现完整的 Transformer 编码器-解码器架构，严格遵循原论文的 Post-LN 结构、正弦位置编码、多头注意力与联合训练设计，采用 **Multi30k 英德翻译数据集** 进行验证，使用行业标准 BLEU 指标评估，完整包含数据预处理、模型训练、自回归推理、指标计算与训练曲线可视化全流程。

## 架构
本次复现严格遵循《Attention Is All You Need》原文的标准架构，整体分为「输入嵌入与位置编码」「堆叠编码器」「堆叠解码器」「输出投影层」四大模块，完整保留原论文的所有核心设计。

- **输入嵌入与位置编码**：输入 token 首先经过词嵌入层映射为 d_model 维向量，并乘以 √d_model 进行幅值缩放；随后叠加**正弦/余弦位置编码**注入序列位置信息，再经过 Dropout 层送入后续网络。位置编码不参与训练，通过三角函数的相对位置特性让模型天然支持长度泛化。
- **编码器（Encoder）**：由 N 个完全相同的编码器层堆叠而成。每层包含两个子层：**多头自注意力层**与**位置前馈网络层**，每个子层均采用「残差连接 + 层归一化（LayerNorm）」的 Post-LN 结构。编码器最终输出整段源序列的上下文特征表示，所有位置的计算完全并行。
- **解码器（Decoder）**：由 N 个完全相同的解码器层堆叠而成。每层包含三个子层：**掩码多头自注意力层**（通过下三角掩码屏蔽未来位置的 token，保证自回归特性）、**编码器-解码器交叉注意力层**（Query 来自解码器输出，Key/Value 来自编码器最终输出，实现源序列到目标序列的信息对齐）、**位置前馈网络层**，三个子层同样采用残差+层归一化结构。
- **多头注意力机制**：将 Query、Key、Value 通过线性投影后拆分为 h 个并行的注意力头，每个头独立计算缩放点积注意力，最后将多个头的输出拼接并做线性投影。多头机制让模型可以同时捕捉不同表示子空间、不同位置的依赖关系，显著提升建模能力。
- **掩码机制**：包含两类掩码，Padding Mask 忽略序列中的填充位置，避免无效 token 参与计算；Look-ahead Mask（下三角掩码）保证解码器在预测第 t 个 token 时，只能看到 t 及之前的位置，严格符合自回归生成的逻辑。
- **输出层与损失**：解码器输出经过线性层投影到目标词表大小，通过 Softmax 得到每个位置的词概率分布。训练时采用教师强制（Teacher Forcing）并行计算所有位置的交叉熵损失，忽略填充位置；推理时采用自回归方式，逐 token 生成输出序列。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/025a0fbde596499ba5e82ca7ff2105d7.png)


**具体网络细节**可以查看我的这篇文章：Transformer深入浅出
**注意**：为降低训练开销、便于快速复现验证，本实现采用轻量化配置（3层堆叠、256维特征、4/8头注意力），数据集使用小型 Multi30k 语料，但模型结构、注意力计算、位置编码、训练范式均严格对齐原论文标准。

## 数据集
本次复现使用 **Multi30k 英德平行翻译数据集**（bentrevett 整理版本），是机器翻译领域经典的小型基准数据集。
- **任务类型**：英语 → 德语 的神经机器翻译任务。
- **数据规模**：训练集 29000 句，验证集 1014 句，测试集 1000 句；所有语料均来自图像描述场景，句式简短、语义明确，适合快速验证模型能力。
- **预处理方式**：采用空格分词 + 小写化处理，分别为源语言和目标语言构建 8000 词大小的词表，包含 `<PAD>`、`<SOS>`、`<EOS>`、`<UNK>` 四个特殊标记，序列最大长度限制为 50，超出部分截断。
- **评估指标**：采用机器翻译领域通用标准指标 **BLEU（Bilingual Evaluation Understudy）**，使用 sacrebleu 工具计算，保证结果可复现、可横向对比。

数据集官方地址：https://huggingface.co/datasets/bentrevett/multi30k
<img width="1265" height="700" alt="image" src="https://github.com/user-attachments/assets/aaa0c8cc-4a74-4ae0-8558-6ec9be0e7070" />

---
## Introduction
Transformer, proposed by **Ashish Vaswani, Noam Shazeer, Niki Parmar et al.** in 2017 (NeurIPS 2017), is a groundbreaking innovation in sequence modeling. Its core contribution is that it **completely abandons recurrent structures and implements sequence-to-sequence modeling solely based on self-attention mechanism**, solving the core pain points of traditional RNN/LSTM models — inability of parallel computation and degradation of long-distance dependency propagation. It has redefined the foundational architecture for almost all sequence tasks including natural language processing, speech, and computer vision.

Before Transformer, mainstream sequence models relied on recurrent neural networks to operate sequentially along time steps, making parallel training impossible and long-distance semantic delivery inefficient. Even with attention mechanisms as auxiliary components, the recurrent structure itself remained a dual bottleneck of performance and speed. For the first time, Transformer takes **self-attention** as the core computing unit, calculates the correlation weights of all positions in the sequence simultaneously through scaled dot-product attention, captures information from different semantic subspaces via multi-head mechanism, and preserves sequence order information with positional encoding. It achieves fully parallel sequence modeling, with both training efficiency and modeling capability improved by orders of magnitude.

On the WMT 2014 English-to-German translation task, the base Transformer achieved **28.4 BLEU**, outperforming all previous RNN and convolutional models, while requiring only 1/4 of the training cost of the previous state-of-the-art model. This reproduction implements the complete Transformer encoder-decoder architecture from scratch based on PyTorch, strictly following the original paper's Post-LN structure, sinusoidal positional encoding, multi-head attention and joint training design. It is validated on the **Multi30k English-German translation dataset** and evaluated using the industry-standard BLEU metric, covering the full pipeline of data preprocessing, model training, autoregressive inference, metric calculation and training curve visualization.

## Architecture
This implementation faithfully follows the standard architecture from *Attention Is All You Need*, consisting of four main modules: input embedding with positional encoding, stacked encoder, stacked decoder, and output projection layer, with all core designs from the original paper preserved.

- **Input Embedding & Positional Encoding**: Input tokens are first mapped to d_model-dimensional vectors via an embedding layer, scaled by √d_model for amplitude adjustment. Then **sinusoidal/cosine positional encoding** is added to inject sequence position information, followed by a Dropout layer before feeding into subsequent networks. Positional encoding is not learnable, and allows the model to naturally generalize to variable sequence lengths through the relative position property of trigonometric functions.
- **Encoder**: Stacked by N identical encoder layers. Each layer contains two sub-layers: a **multi-head self-attention layer** and a **position-wise feed-forward network layer**. Each sub-layer adopts the Post-LN structure of "residual connection + LayerNorm". The encoder finally outputs contextual feature representations of the entire source sequence, with all positions computed fully in parallel.
- **Decoder**: Stacked by N identical decoder layers. Each layer contains three sub-layers: a **masked multi-head self-attention layer** (masking future tokens via a lower-triangular mask to guarantee autoregressive property), an **encoder-decoder cross-attention layer** (Queries from decoder output, Keys/Values from final encoder output, achieving information alignment from source to target sequence), and a **position-wise feed-forward network layer**. All three sub-layers also adopt residual + layer normalization structure.
- **Multi-Head Attention**: Query, Key and Value are linearly projected and split into h parallel attention heads. Each head independently computes scaled dot-product attention, and outputs of all heads are finally concatenated and linearly projected. The multi-head mechanism allows the model to simultaneously capture dependencies from different representation subspaces and different positions, significantly improving modeling capability.
- **Mask Mechanism**: Two types of masks are included. Padding Mask ignores padding positions in the sequence to avoid invalid tokens participating in computation. Look-ahead Mask (lower-triangular mask) ensures that when predicting the t-th token, the decoder can only see positions up to t, strictly conforming to the logic of autoregressive generation.
- **Output Layer & Loss**: The decoder output is projected to the target vocabulary size via a linear layer, and word probability distribution at each position is obtained through Softmax. During training, teacher forcing is used to compute cross-entropy loss at all positions in parallel, ignoring padding positions. During inference, output sequences are generated token by token in an autoregressive manner.

**Note**: To reduce training cost and facilitate quick reproduction and verification, this implementation uses a lightweight configuration (3 stacked layers, 256-dimensional features, 4/8 attention heads) and the small Multi30k corpus. However, model structure, attention computation, positional encoding, and training paradigm are all strictly aligned with the original paper standard.

## Dataset
This reproduction uses the **Multi30k English-German parallel translation dataset** (curated by bentrevett), a classic small benchmark dataset in the field of machine translation.
- **Task**: English → German neural machine translation.
- **Data Scale**: 29,000 training sentences, 1,014 validation sentences, and 1,000 test sentences. All corpus comes from image captioning scenarios, with short sentences and clear semantics, suitable for quickly verifying model capability.
- **Preprocessing**: Tokenized by whitespace with lowercasing. Vocabularies of size 8000 are built for source and target languages respectively, including four special tokens: `<PAD>`, `<SOS>`, `<EOS>`, `<UNK>`. Maximum sequence length is limited to 50, with longer sequences truncated.
- **Evaluation Metric**: **BLEU (Bilingual Evaluation Understudy)**, the universal standard metric for machine translation, computed using the sacrebleu toolkit to ensure reproducible and comparable results.

Official dataset page: https://huggingface.co/datasets/bentrevett/multi30k

---
## 原文章 | Original article
Vaswani A, Shazeer N, Parmar N, et al. Attention Is All You Need[C]//Advances in Neural Information Processing Systems. 2017: 5998-6008.

# Faster R-CNN  
### 选择语言 | Language  
[中文简介](#简介) | [English](#introduction)

### 结果 | Result  
<img width="1268" height="586" alt="loss_curve" src="https://github.com/user-attachments/assets/30ef8be9-2723-4667-b184-7a48653bc6fe" />
<img width="1268" height="586" alt="map_curve" src="https://github.com/user-attachments/assets/c01a7eac-9292-4f57-805a-1908f490fefe" />
<img width="723" height="985" alt="test_detect" src="https://github.com/user-attachments/assets/dbc3e484-6daf-410c-8b91-c2ca04bc0dbf" />

---

## 简介  
Faster R-CNN 由 **Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun** 于 2015 年提出（NIPS 2015，TPAMI 2017），是针对 Fast R-CNN 仍依赖外部独立算法生成候选区域（如选择性搜索）这一痛点的质变性革新。其核心贡献在于**区域建议网络（Region Proposal Network, RPN）**，将候选框生成嵌入全卷积网络，与下游检测共享底层特征，首次实现了候选生成、特征提取、分类与回归的全链路端到端联合训练，标志着两阶段检测器走向成熟的实时化。  

Fast R-CNN 虽然将特征提取变为一次性全局卷积，但仍需在 GPU 外部运行选择性搜索，生成约 2000 个候选区域，该步骤成为推理速度的新瓶颈。Faster R-CNN 将候选生成完全交由网络自身完成：RPN 在共享的卷积特征图上滑动窗口，每个位置同时预测多个尺度/长宽比的锚框（anchor）的 **“目标/背景”二分类得分** 与 **初始边界框回归偏移**。随后，筛选后的候选框经由 RoI Pooling 提取固定维度特征，送入与 Fast R-CNN 一致的分类与边界框回归多任务分支。  

在 PASCAL VOC 2007 上，Faster R-CNN 以 VGG-16 为骨干取得了 **73.2% mAP**，而推理速度可达 **5 fps**（未优化版本）～**17 fps**（共享特征），彻底打破了候选生成的速度枷锁。本复现中，为进一步压缩模型体积与推理成本，我们采用 **MobileNet V3‑Large + FPN** 替代传统 VGG/ResNet，并保留 RPN、RoI Pooling 及多任务联合训练等全部核心结构，在小样本数据集上快速迭代学习。  


## 架构  
本次复现严格遵循原始 Faster R-CNN 标准架构，整体分为「共享卷积骨干与特征金字塔」「区域建议网络（RPN）」「RoI Pooling 与区域特征提取」「多任务分类/回归联合训练」四大模块，并采用 **MobileNet V3‑Large + FPN** 作为骨干网络，兼顾轻量与多尺度表达能力。

- **共享卷积骨干与 FPN**：输入整张图像后，由 **MobileNet V3‑Large** 提取多层卷积特征，再通过**特征金字塔网络（FPN）** 融合高层语义与低层细节，输出不同分辨率的特征图 \(\{P_2, P_3, P_4, P_5\}\)。所有后续模块均共享这些特征图，彻底消除重复计算。  
- **区域建议网络（RPN）**：在每一层 FPN 特征图上施加一个全卷积子网络，各位置关联多个预设大小和比例的锚框（anchor）。RPN 同时输出**前景/背景得分**与**锚框修正偏移**，经过非极大值抑制（NMS）后保留最可能的候选区域（proposals），送入下一阶段。该模块完全由网络端到端学习，无需任何外部算法。  
- **RoI Pooling 与区域特征提取**：将 RPN 生成的候选区域映射到对应层级的 FPN 特征图上，通过 **RoI Pooling**（本实现使用 torchvision 内置算子）自适应池化为固定尺寸的特征向量，保证全连接层的输入维度统一。  
- **多任务分支与联合训练**：  
  - **分类分支**：Softmax 输出候选框在所有类别（含背景）上的概率分布，替代传统 SVM。  
  - **边界框回归分支**：对每一类预测坐标精修偏移量 (tx, ty, tw, th)。  
  - **损失函数**：联合优化 RPN 的二分类损失（物体/非物体）与回归损失，以及最终检测头的分类损失与回归损失，1 个网络整体通过反向传播同步更新。  
- **推理后处理**：推理时 RPN 生成 proposals，RoI Pooling 提取特征，双分支输出类别置信度与精修框坐标，最后对每个类应用 NMS 去除高重叠检测框，得到最终结果。



每个中心都有九个的滑动窗口：
<img width="1536" height="705" alt="image" src="https://github.com/user-attachments/assets/af619327-d479-4ced-b121-c62671ab9755" />
<img width="1360" height="539" alt="image" src="https://github.com/user-attachments/assets/8644459d-7136-4b61-a035-2d10cd53a03f" />

候选框数量：
<img width="1520" height="650" alt="image" src="https://github.com/user-attachments/assets/a06dc4ad-4b65-4b30-8668-ea76b86e7492" />
网络架构：
<img width="774" height="431" alt="image" src="https://github.com/user-attachments/assets/b81c8f3d-abd9-44ef-8a38-c7bbad32db1f" />
<img width="675" height="441" alt="image" src="https://github.com/user-attachments/assets/4c97b6b6-d47b-4b39-9ad7-77808ee90501" />


Loss：
<img width="1529" height="700" alt="image" src="https://github.com/user-attachments/assets/95b18389-4339-444d-82ad-4993190e9b01" />
<img width="1558" height="693" alt="image" src="https://github.com/user-attachments/assets/0b945127-85d7-4d35-b620-bed7f9fb8471" />
<img width="1564" height="705" alt="image" src="https://github.com/user-attachments/assets/23b1e711-8c9b-4f85-a0fa-8f8edbd69227" />
<img width="1505" height="735" alt="image" src="https://github.com/user-attachments/assets/e929263a-80a6-4a29-8f99-caf143910b8d" />

**注意**：为降低计算开销、便于快速验证，本实现仅使用 **VOC 2007 子集**（bird, cat, dog 三类），骨干替换为 MobileNet V3‑Large + FPN，但 RPN 的多尺度锚框策略、RoI Pooling、四任务联合损失等核心设计均严格遵循原文。

## 数据集  
沿用 PASCAL VOC 2007 自定义子集，仅保留 **bird（鸟）、cat（猫）、dog（狗）** 三个类别。  
- **数据来源**：完整 VOC 2007 train/val 集中筛选出包含该三类目标的图像，共 1000 张及对应标注。  
- **标注格式**：PASCAL VOC XML 格式，记录目标类别和边界框坐标 (xmin, ymin, xmax, ymax)。  
- **数据划分**：按 8:2 随机划分为训练集和验证集，用于模型训练、mAP 评估与可视化分析。  

数据集官方地址：http://host.robots.ox.ac.uk/pascal/VOC/voc2007/  

---

## Introduction  
Faster R-CNN, proposed by **Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun** in 2015 (NIPS 2015, TPAMI 2017), is a landmark improvement over Fast R-CNN. By introducing the **Region Proposal Network (RPN)**, it eliminates the last external, non‑trainable component of the object‑detection pipeline — the region proposal algorithm (e.g., Selective Search) — and achieves **fully end‑to‑end training** of a unified network that generates proposals, extracts features, classifies objects, and refines bounding boxes.

Fast R‑CNN had already made detection much faster by extracting global convolutional features only once; however, proposal generation still ran on CPU and became the new bottleneck. Faster R‑CNN solves this by sharing convolutional features between the RPN and the detection head. The RPN slides small networks over the shared feature maps, simultaneously predicting **objectness scores** and **bounding‑box adjustments** for multiple anchor boxes per location. The high‑quality proposals are then pooled (e.g., via RoI Pooling or RoI Align) and fed to the detection sub‑network for final classification and regression.

Using a VGG‑16 backbone, Faster R‑CNN achieved **73.2% mAP** on PASCAL VOC 2007 with inference speeds of **5 fps** (non‑shared features) to **17 fps** (shared), opening the door for real‑time two‑stage detectors. In this reproduction, we further explore **lightweight deployment** by adopting **MobileNetV3‑Large + FPN** as the backbone, keeping the RPN, RoI Pooling, and multi‑task training intact while adapting to a small custom subset.


## Architecture  
This implementation faithfully follows the standard Faster R‑CNN pipeline, consisting of four main modules: shared backbone with FPN, Region Proposal Network, RoI Pooling extraction, and multi‑task classification/regression training. The backbone is set to **MobileNetV3‑Large + FPN** for enhanced efficiency and multi‑scale feature fusion.

- **Shared Backbone & FPN**: The entire image passes through MobileNetV3‑Large, generating multiple feature levels, which are then merged by a **Feature Pyramid Network** to produce semantically strong, multi‑scale feature maps \(\{P_2, P_3, P_4, P_5\}\). All subsequent stages reuse these maps, eliminating redundant computation.  
- **Region Proposal Network (RPN)**: A fully convolutional sub‑network is applied to each FPN level. At each sliding‑window location, it predicts **objectness** (foreground/background) and **bounding‑box regressors** for a set of predefined anchor boxes. After non‑maximum suppression (NMS), the top‑scoring proposals are passed to the next stage.  
- **RoI Pooling & Feature Extraction**: Proposals are mapped to the appropriate FPN level based on their size. An **RoI Pooling** layer (torchvision implementation) then max‑pools each region into a fixed‑size feature tensor, compatible with the subsequent fully‑connected heads.  
- **Multi‑task Training**: The network heads split into two parallel branches: a **Softmax classifier** for C+1 categories (including background) and a **bounding‑box regressor** that refines deltas for each class. The loss function combines four components: RPN classification loss, RPN regression loss, detection classification loss, and detection regression loss. All are jointly optimized end‑to‑end.  
- **Inference**: At test time, the RPN generates proposals, RoI Pooling extracts features, the dual heads output class probabilities and refined boxes, and class‑wise NMS removes duplicates to produce final detections.

**Note**: To facilitate rapid experimentation, we restrict the dataset to the VOC 2007 subset (bird, cat, dog). The backbone is replaced by MobileNetV3‑Large + FPN. All core designs — RPN anchors, RoI Pooling, multi‑task joint loss — remain strictly consistent with the original paper.

## Dataset  
We use a custom subset of PASCAL VOC 2007 containing only **bird, cat, and dog**.  
- **Source**: Filtered from the official VOC 2007 train/val set, comprising 1000 images with corresponding XML annotations.  
- **Annotation format**: PASCAL VOC XML, storing class labels and bounding box coordinates (xmin, ymin, xmax, ymax).  
- **Split**: Randomly divided into 80% training and 20% validation for model training, evaluation, and visualization.  

Official dataset page: http://host.robots.ox.ac.uk/pascal/VOC/voc2007/  

---
## 原文章 | Original article  
Ren S, He K, Girshick R, et al. Faster R‑CNN: Towards Real‑Time Object Detection with Region Proposal Networks[J]. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 2017, 39(6): 1137‑1149.  
(Also published in *Advances in Neural Information Processing Systems*, 2015.)

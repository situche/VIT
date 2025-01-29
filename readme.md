# Vision Transformer (ViT) 实现

这是一个基于 PyTorch 实现的 Vision Transformer (ViT) 模型，ViT 将 Transformer 架构应用于图像分类任务。

## 特性

- **Transformer 编码器**：ViT 的核心是 Transformer 编码器层，通过自注意力机制（Self-Attention）处理图像块。
- **CLS Token 用于分类**：在输入序列的最前面加入一个特殊的分类 token（CLS token），用于图像分类任务。
- **位置嵌入**：模型使用位置嵌入来保留图像块的空间顺序。
- **灵活性**：模型可以选择进行分类任务，或者返回原始的 Transformer 输出。

## 依赖项

该模型使用 PyTorch 实现，并且使用了 `einops` 库进行张量操作。

安装所需的依赖项，可以使用以下命令：

```
pip install torch einops
```

## 模型概述

ViT 模型的工作流程如下：

1. **图像切分为块（Patch）**：首先将图像切分为多个小块（patch），每个块的大小为 `path_dim x path_dim`。
2. **线性投影**：将每个图像块展平后，通过线性层将其映射到指定的维度空间。
3. **位置嵌入**：为每个图像块添加位置编码，保持空间结构的信息。
4. **Transformer 编码器**：将所有的图像块（包括 CLS token）作为输入，经过 Transformer 编码器进行处理，得到最终的特征表示。
5. **分类任务**：如果是分类任务，通过分类头（MLP）将 CLS token 的输出转换为类别标签。

## 如何使用

### 模型初始化

首先，创建 ViT 模型对象，并提供所需的参数：

```python
from vit import VIT

# 设置模型参数
model = VIT(img_dim=224, path_dim=16, in_channels=3, num_classes=10, dim=512, blocks=6, heads=4, dim_linear_block=1024, dropout=0.1, classification=True)
```

### 模型前向传播

使用模型进行前向传播，输入图像和可选的掩码（mask）：

```python
# 假设输入图像是一个形状为 (batch_size, channels, height, width) 的张量
img = torch.randn(32, 3, 224, 224)  # batch_size=32, 图像尺寸为 224x224

# 前向传播
output = model(img)
```

如果你选择了分类任务，`output` 将是一个形状为 `(batch_size, num_classes)` 的张量，表示每个样本的预测类别。

### 非分类任务

如果不进行分类，只需设置 `classification=False`，模型会返回 Transformer 编码器的原始输出：

```python
model_no_classification = VIT(img_dim=224, path_dim=16, in_channels=3, num_classes=10, dim=512, blocks=6, heads=4, dim_linear_block=1024, dropout=0.1, classification=False)

# 前向传播
output = model_no_classification(img)
```

此时，`output` 的形状为 `(batch_size, num_patches, dim)`，即每个图像块的编码表示。

## 示例

```python
img_dim = 32
path_dim = 4
in_channels = 3

batch_size = 8
image = torch.randn(batch_size, in_channels, img_dim, img_dim)

model = VIT(img_dim=img_dim, path_dim=path_dim)
output = model(image)
print('输出形状', output.shape)

# 输出
输出形状 torch.Size([8, 10])
```

## 参数说明

- `img_dim`: 输入图像的尺寸（假设为正方形图像）。
- `path_dim`: 每个图像块的尺寸（正方形块）。
- `in_channels`: 输入图像的通道数，默认值为 3（RGB 图像）。
- `num_classes`: 分类任务中的类别数。
- `dim`: Transformer 中间的隐藏维度。
- `blocks`: Transformer 编码器的层数。
- `heads`: 每个 Transformer 层中多头注意力机制的头数。
- `dim_linear_block`: Transformer 中前馈网络的维度。
- `dropout`: Dropout 比例，防止过拟合。
- `classification`: 是否进行分类任务。如果为 `True`，则模型将输出分类结果；如果为 `False`，则返回 Transformer 的编码输出。

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

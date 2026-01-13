# RF模块使用文档

## 概述

RF（Rectified Flow）模块是一个可插拔的embedding生成器，可以轻松集成到任何推荐系统中以增强embedding质量。本文档说明如何在您的模型中使用RF模块。

## 特性

- **即插即用**：3步集成到任何推荐模型
- **独立优化**：RF有自己的优化器和学习率
- **灵活配置**：所有参数都可通过config控制
- **可选使用**：通过 `use_rf`开关轻松启用/禁用
- **Warmup支持**：渐进式启用RF，避免训练初期的不稳定

## 快速开始

### 1. 初始化RF模块

```python
from models.rf_modules import RFEmbeddingGenerator

class MyRecommender(BaseRecommender):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # 其他网络组件初始化...

        # 添加RF生成器（可选）
        if config["use_rf"] if "use_rf" in config else False:
            self.rf_generator = RFEmbeddingGenerator(
                # 必需参数
                embedding_dim=self.embedding_dim,

                # 可选参数（不配置则使用默认值）
                hidden_dim=config["rf_hidden_dim"] if "rf_hidden_dim" in config else 256,
                learning_rate=config["rf_lr"] if "rf_lr" in config else 0.001,
            )
```

### 2. 在forward中使用

```python
def forward(self, adj, train=False):
    # 1. 计算原始embeddings
    original_embeds = ...  # 你的原始网络计算

    # 2. 计算条件embeddings（RF的输入）
    image_cond = ...
    text_cond = ...

    # 3. 使用RF增强
    if hasattr(self, 'rf_generator'):
        if train:
            # 训练：自动计算损失并更新RF参数
            loss_dict = self.rf_generator.compute_loss_and_step(
                target_embeds=original_embeds.detach(),
                conditions=[image_cond.detach(), text_cond.detach()],
            )

            # 生成RF embeddings
            rf_embeds = self.rf_generator.generate([image_cond, text_cond])

            # 混合原始和RF生成的embeddings
            final_embeds = self.rf_generator.mix_embeddings(
                original_embeds, rf_embeds.detach(), training=True
            )
        else:
            # 推理：直接生成并混合
            rf_embeds = self.rf_generator.generate([image_cond, text_cond])
            final_embeds = self.rf_generator.mix_embeddings(
                original_embeds, rf_embeds, training=False
            )
    else:
        final_embeds = original_embeds

    return final_embeds
```

### 3. 配置文件

```yaml
# model_config.yaml

# 启用RF
use_rf: True

# RF参数（可选，不配置则使用默认值）
rf_hidden_dim: 128
rf_n_layers: 2
rf_lr: 0.0001
rf_sampling_steps: 10
rf_warmup_epochs: 5
rf_mix_ratio: 0.1           # 训练时混合比例
rf_inference_mix_ratio: 0.2 # 推理时混合比例
```

## API详细说明

### RFEmbeddingGenerator

#### 初始化参数

| 参数                      | 类型  | 默认值         | 说明             |
| ------------------------- | ----- | -------------- | ---------------- |
| `embedding_dim`         | int   | **必需** | Embedding维度    |
| `hidden_dim`            | int   | 256            | 隐藏层维度       |
| `n_layers`              | int   | 2              | 网络层数         |
| `dropout`               | float | 0.1            | Dropout率        |
| `learning_rate`         | float | 0.001          | 独立学习率       |
| `sampling_steps`        | int   | 10             | ODE采样步数      |
| `user_guidance_scale`   | float | 0.2            | 用户先验缩放因子 |
| `guidance_decay_power`  | float | 2.0            | 先验衰减指数     |
| `cosine_guidance_scale` | float | 0.1            | 余弦梯度缩放因子 |
| `cosine_decay_power`    | float | 2.0            | 余弦梯度衰减指数 |
| `warmup_epochs`         | int   | 0              | Warmup epoch数   |
| `train_mix_ratio`       | float | 0.5            | 训练混合比例     |
| `inference_mix_ratio`   | float | 0.5            | 推理混合比例     |
| `contrast_temp`         | float | 0.2            | 对比损失温度     |
| `contrast_weight`       | float | 1.0            | 对比损失权重     |

#### 核心方法

##### `compute_loss_and_step()`

计算RF损失并执行优化步骤。

**参数：**

- `target_embeds` (Tensor): 目标embedding，shape (batch, embedding_dim)
- `conditions` (List[Tensor]): 条件embeddings列表，会自动拼接
- `user_prior` (Optional[Tensor]): 可选的用户先验指导
- `epoch` (Optional[int]): 当前epoch（用于warmup控制）

**返回：**

- `loss_dict` (Dict[str, float]): 包含 "rf_loss", "cl_loss", "total_loss"

**示例：**

```python
loss_dict = rf_gen.compute_loss_and_step(
    target_embeds=target.detach(),
    conditions=[image_cond.detach(), text_cond.detach()],
    user_prior=prior.detach(),
    epoch=10,
)
print(f"RF Loss: {loss_dict['rf_loss']:.6f}")
```

##### `generate()`

使用ODE采样生成embeddings。

**参数：**

- `conditions` (List[Tensor]): 条件embeddings列表
- `n_steps` (Optional[int]): ODE采样步数（None则使用默认值）

**返回：**

- `generated_embeds` (Tensor): 生成的embeddings

**示例：**

```python
rf_embeds = rf_gen.generate([image_cond, text_cond], n_steps=10)
```

##### `mix_embeddings()`

混合原始和生成的embeddings。

**参数：**

- `original_embeds` (Tensor): 原始embedding
- `generated_embeds` (Tensor): RF生成的embedding
- `training` (bool): 训练/推理模式
- `epoch` (Optional[int]): 当前epoch

**返回：**

- `mixed_embeds` (Tensor): 混合后的embeddings

**说明：**

- Warmup期间（epoch < warmup_epochs）返回纯原始embeddings
- Warmup后根据mix_ratio混合

**示例：**

```python
# 训练模式
mixed = rf_gen.mix_embeddings(original, generated, training=True, epoch=10)

# 推理模式
mixed = rf_gen.mix_embeddings(original, generated, training=False)
```

##### `set_epoch()`

设置当前epoch（由trainer调用）。

**参数：**

- `epoch` (int): 当前epoch

**示例：**

```python
rf_gen.set_epoch(epoch)
```

## 高级功能

### 1. 用户先验指导

RF模块支持使用用户特定的先验知识来指导生成过程：

```python
# 计算用户先验
user_specific = ...  # 用户特定的特征
user_general = ...   # 通用特征（平均值）
user_prior = user_specific - user_general

# 使用先验指导训练
loss_dict = rf_gen.compute_loss_and_step(
    target_embeds=target,
    conditions=[cond1, cond2],
    user_prior=user_prior,  # 添加用户先验
)
```

### 2. Warmup策略

使用warmup可以让模型先收敛，然后再启用RF：

```python
# 配置文件
rf_warmup_epochs: 5  # 前5个epoch不使用RF混合

# 训练循环
for epoch in range(epochs):
    rf_gen.set_epoch(epoch)
    # 前5个epoch：RF训练但不参与混合（返回原始embeddings）
    # 第6个epoch开始：RF参与混合
```

### 3. 不同训练/推理混合比例

训练和推理时可以使用不同的混合比例：

```python
# 配置
rf_mix_ratio: 0.1           # 训练时：90%原始 + 10%RF
rf_inference_mix_ratio: 0.2 # 推理时：80%原始 + 20%RF

# 自动根据training参数选择
mixed = rf_gen.mix_embeddings(original, generated, training=True)  # 使用0.1
mixed = rf_gen.mix_embeddings(original, generated, training=False) # 使用0.2
```

## 完整示例：在GUME中集成RF

参考 `src/models/rfgume.py` 查看完整的集成示例。

关键代码片段：

```python
class RFGUME(GUME):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # 初始化RF生成器
        if config["use_rf"] if "use_rf" in config else True:
            self.rf_generator = RFEmbeddingGenerator(
                embedding_dim=self.embedding_dim,
                hidden_dim=config["rf_hidden_dim"] if "rf_hidden_dim" in config else 256,
                # ... 其他参数
            )

    def forward(self, adj, train=False):
        # 1. 原始GUME计算
        extended_id_embeds_target = self.conv_ui(adj, user_embeds, item_embeds)

        # 2. 计算条件
        explicit_image_embeds = ...
        explicit_text_embeds = ...

        # 3. RF训练和生成
        if self.use_rf and train:
            # 计算用户先验
            user_prior = ...

            # RF训练
            loss_dict = self.rf_generator.compute_loss_and_step(
                target_embeds=extended_id_embeds_target.detach(),
                conditions=[explicit_image_embeds.detach(), explicit_text_embeds.detach()],
                user_prior=user_prior.detach(),
            )

            # 生成并混合
            rf_embeds = self.rf_generator.generate([explicit_image_embeds, explicit_text_embeds])
            extended_id_embeds = self.rf_generator.mix_embeddings(
                extended_id_embeds_target, rf_embeds.detach(), training=True
            )
        elif self.use_rf and not train:
            # 推理模式
            rf_embeds = self.rf_generator.generate([explicit_image_embeds, explicit_text_embeds])
            extended_id_embeds = self.rf_generator.mix_embeddings(
                extended_id_embeds_target, rf_embeds, training=False
            )
        else:
            extended_id_embeds = extended_id_embeds_target

        # 4. 继续GUME的其余计算
        ...
```

## 参数调优建议

### 基础参数

- **embedding_dim**: 必须与原始模型的embedding维度一致
- **hidden_dim**: 推荐值 [128, 256, 512]，更大的值可能提高表达能力但增加计算成本
- **n_layers**: 推荐值 [2, 3]，通常2层就足够

### 训练参数

- **learning_rate**: 推荐值 [0.0001, 0.001]

  - 比主模型的学习率稍小通常效果更好
  - 如果RF损失不收敛，尝试增大学习率
- **sampling_steps**: 推荐值 [5, 10, 20]

  - 更多步数生成质量更好，但速度更慢
  - 训练时可以用较少步数（5-10），推理时用更多步数

### 混合策略

- **warmup_epochs**: 推荐值 [0, 5, 10]

  - 0: 从一开始就启用RF
  - 5-10: 让基础模型先预热
- **train_mix_ratio**: 推荐值 [0.1, 0.2, 0.3]

  - 小值（0.1）: RF作为辅助，主要依赖原始embedding
  - 大值（0.5+）: RF和原始embedding平等混合
- **inference_mix_ratio**: 推荐值 [0.2, 0.3, 0.5]

  - 通常比训练时稍大
  - 可以通过验证集调优

### 先验指导

- **user_guidance_scale**: 推荐值 [0.1, 0.2, 0.5]

  - 控制用户个性化指导的强度
  - 如果用户个性化重要，可以增大
- **cosine_guidance_scale**: 推荐值 [0.05, 0.1, 0.2]

  - 控制与目标的余弦相似度指导强度
  - 有助于生成更接近目标的embeddings

## 常见问题

### Q1: RF模块会增加多少训练时间？

A: 通常增加10-20%的训练时间，取决于 `sampling_steps`和网络大小。

### Q2: 需要调整原始模型的学习率吗？

A: 不需要。RF有独立的优化器和学习率，不影响原始模型。

### Q3: 什么时候应该使用warmup？

A: 如果发现训练初期不稳定或损失震荡，建议使用warmup（5-10 epochs）。

### Q4: 如何知道RF是否有效？

A: 观察以下指标：

- RF损失应该逐渐下降
- 对比损失（cl_loss）应该保持在合理范围
- 最终推荐指标（如NDCG, Recall）应该有提升

### Q5: 可以只在推理时使用RF吗？

A: 不行。RF需要训练才能学会生成有效的embeddings。

## 故障排除

### 问题1: RF损失不收敛

**解决方案：**

- 增大 `learning_rate`
- 减少 `sampling_steps`（训练时）
- 检查条件embeddings是否合理（不应包含NaN或Inf）

### 问题2: 显存不足

**解决方案：**

- 减小 `hidden_dim`
- 减小 `n_layers`
- 减小 `sampling_steps`
- 使用gradient checkpointing（需要修改代码）

### 问题3: RF没有提升性能

**解决方案：**

- 检查warmup设置是否合理
- 尝试不同的 `mix_ratio`
- 调整 `learning_rate`
- 确保条件embeddings质量足够好

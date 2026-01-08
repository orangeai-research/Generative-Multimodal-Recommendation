# RF-GUME: Rectified Flow for GUME

## 概述

RF-GUME 将 Rectified Flow 方法集成到 GUME 模型中，用于生成 `extended_id_embeds`。该实现采用低耦合设计，便于进行消融实验。

## 核心思想

### 1. 问题定义
在原始 GUME 中，`extended_id_embeds` 是通过图卷积直接计算得到的：
```python
extended_id_embeds = self.conv_ui(adj, user_embeds, item_embeds)
```

### 2. RF-GUME 的改进
使用 Rectified Flow 从噪声和多模态条件生成 `extended_id_embeds`：

**条件（Conditions）**：
- `explicit_image_item`: 经过图像模态图卷积的物品嵌入
- `explicit_text_item`: 经过文本模态图卷积的物品嵌入

**生成过程**：
```
Z_0 (噪声) + Conditions (image + text)
    ↓ [ODE求解: dZ_t = v(Z_t, t, conditions) dt]
Z_1 (extended_id_embeds)
```

## 训练流程

### 训练阶段（重要修正）

**正确的训练方式**：
```python
# 1. 获取目标 embeds（原始GUME的输出）
extended_id_embeds_target = self.conv_ui(adj, user_embeds, item_embeds)

# 2. 使用RF生成embeds（这样才有梯度！）
rf_user_embeds = self.rf_user_generator(
    explicit_image_user,
    explicit_text_user,
    n_steps=sampling_steps
)
rf_item_embeds = self.rf_item_generator(
    explicit_image_item,
    explicit_text_item,
    n_steps=sampling_steps
)

# 3. 使用RF生成的embeds进行后续计算
extended_id_embeds = torch.cat([rf_user_embeds, rf_item_embeds], dim=0)
# 然后用 extended_id_embeds 继续GUME的剩余部分
```

**三个损失项**：

1. **RF Loss（核心）**：学习速度场 v(x, t, c)
   ```python
   # 对于随机采样的 t ~ Uniform[0,1]
   X_t = t * X1 + (1-t) * X0  # X1是目标，X0是噪声
   v_pred = velocity_net(X_t, t, conditions)
   v_target = X1 - X0  # 直线方向
   rf_loss = ||v_pred - v_target||^2
   ```

2. **Contrast Loss（约束）**：确保生成质量
   ```python
   contrast_loss = MSE(rf_generated, target)
   ```

3. **GUME Loss（原始损失）**：BPR + 对齐损失 + 正则化

**总损失**：
```python
total_loss = gume_loss + λ_rf * rf_loss + λ_contrast * contrast_loss
```

### 推理阶段
```python
# 使用RF生成器从噪声生成embeds
z_0 = torch.randn(batch_size, embedding_dim)
extended_id_embeds = rf_generator.sample_ode(
    z_0,
    image_cond,
    text_cond,
    n_steps=sampling_steps
)
```

## 架构设计

### 低耦合设计
```
GUME_RF
├── rf_user_generator (RFExtendedIdGenerator)
│   └── velocity_net (SimpleVelocityNet)
└── rf_item_generator (RFExtendedIdGenerator)
    └── velocity_net (SimpleVelocityNet)
```

**优势**：
- RF 模块独立，易于添加/移除
- 可以单独训练 RF 生成器
- 便于消融实验（设置 `use_rf=False` 即可回退到原始GUME）

## 配置参数

创建配置文件 `configs/model/RF_GUME.yaml`：

```yaml
# 继承GUME的基础配置
embedding_size: 64

# GUME原始参数
bm_loss: 0.1
um_loss: 0.1
vt_loss: 0.1
reg_weight_1: 1e-4
reg_weight_2: 1e-4
bm_temp: 0.2
um_temp: 0.2
n_ui_layers: 2
n_layers: 2
knn_k: 10

# RF相关参数
use_rf: True                    # 是否使用RF生成extended_id_embeds
rf_hidden_dim: 256             # RF速度场网络的隐藏层维度
rf_n_layers: 3                 # RF速度场网络的层数
rf_dropout: 0.1                # RF网络的dropout率
rf_loss_weight: 1.0            # RF损失的权重
contrast_loss_weight: 0.1      # 对比损失的权重
rf_sampling_steps: 10          # ODE求解步数（训练和推理）
```

## 使用方法

### 训练
```python
from models.rfGume import GUME_RF

# 初始化模型
model = GUME_RF(config, dataset)

# 训练（自动使用RF生成embeds）
loss = model.calculate_loss(interaction)
loss.backward()
optimizer.step()
```

### 消融实验

**实验1：不使用RF（回退到原始GUME）**
```yaml
use_rf: False
```

**实验2：调整RF采样步数**
```yaml
rf_sampling_steps: 1   # 1步生成（快速但可能质量较低）
rf_sampling_steps: 10  # 10步生成（平衡）
rf_sampling_steps: 100 # 100步生成（高质量但慢）
```

**实验3：调整损失权重**
```yaml
rf_loss_weight: 0.5        # 减小RF损失权重
contrast_loss_weight: 0.5  # 增大对比损失权重
```

**实验4：不同的网络结构**
```yaml
rf_hidden_dim: 128   # 小网络
rf_n_layers: 2

rf_hidden_dim: 512   # 大网络
rf_n_layers: 4
```

## 关键实现细节

### 1. 为什么训练时也要用RF生成的embeds？

**错误做法**：
```python
# ❌ 这样RF生成器没有梯度，无法学习
if train:
    extended_id_embeds = extended_id_embeds_target  # 直接用目标
```

**正确做法**：
```python
# ✓ RF生成器参与前向传播，有梯度回传
if train:
    rf_embeds = rf_generator(conditions)  # 生成
    extended_id_embeds = rf_embeds        # 使用生成的
    # 然后计算损失，梯度可以回传到rf_generator
```

### 2. 梯度流动

```
噪声 Z_0
  ↓
velocity_net(Z_t, t, conditions) → extended_id_embeds
  ↓
GUME后续计算 → all_embeds
  ↓
BPR Loss / 对齐损失
  ↓ (梯度回传)
更新 velocity_net 参数
```

同时：
```
target_embeds ← conv_ui(原始GUME)
  ↓
RF Loss = ||v_pred - (target - noise)||^2
  ↓ (梯度回传)
更新 velocity_net 参数
```

### 3. 对比损失的作用

在训练早期，RF生成的embeds可能与目标相差很大。对比损失提供额外的监督信号：
```python
contrast_loss = MSE(rf_generated, gume_target)
```

这有助于：
- 加速收敛
- 确保生成的embeds在合理的范围内
- 作为RF Loss的补充约束

## Reflow（可选扩展）

如果想进一步提升性能，可以实现 Reflow：

```python
# 第1次训练：学习从噪声到目标的流
model_1 = train_rf_gume(data)

# 生成新数据对
Z0, Z1 = generate_pairs_from_model(model_1)

# 第2次训练：在新数据上重新训练（路径会更直）
model_2 = train_rf_gume(new_data=(Z0, Z1))

# 2-rectified flow 可以用更少步数生成高质量embeds
```

## 预期效果

1. **生成质量**：RF生成的embeds应该接近原始GUME的embeds
2. **推荐性能**：应该达到或略优于原始GUME
3. **灵活性**：可以通过调整采样步数权衡速度和质量
4. **可解释性**：条件（image + text）显式地引导生成过程

## 调试建议

### 检查RF Loss
```python
# RF loss应该逐渐下降
print(f"RF Loss: {rf_loss.item():.4f}")
# 初期可能在 0.5-1.0，训练后应降到 0.01-0.1
```

### 检查对比损失
```python
# 对比损失也应该下降
print(f"Contrast Loss: {contrast_loss.item():.4f}")
# 如果一直很高（>0.5），说明RF生成的embeds质量不好
```

### 检查生成的embeds
```python
# 查看生成的embeds是否合理
print(f"Generated embeds - mean: {rf_embeds.mean():.4f}, std: {rf_embeds.std():.4f}")
print(f"Target embeds - mean: {target_embeds.mean():.4f}, std: {target_embeds.std():.4f}")
# 均值和标准差应该接近
```

### 可视化（可选）
```python
# 使用t-SNE可视化生成的embeds和目标embeds
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2)
# 如果分布相似，说明RF学习得好
```

## 参考文献

Liu et al. "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow" arXiv 2022

Wei et al. "GUME: Graphs and User Modalities Enhancement for Long-Tail Multimodal Recommendation" CIKM 2024

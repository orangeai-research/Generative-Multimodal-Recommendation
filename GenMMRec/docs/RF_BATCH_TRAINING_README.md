# RF Models 批量训练使用指南

本文档说明如何使用批量训练脚本来运行和对比 RFLGMRec、RFBM3、RFFREEDOM 三个模型。

## 概述

提供了两种方式进行批量训练：

1. **Python 脚本** (`src/run_rf_models.py`) - 功能完整，支持所有配置
2. **Bash 脚本** (`run_rf_comparison.sh`) - 简化的命令行接口，便于快速使用

## 方式一：使用 Python 脚本

### 基本用法

```bash
cd GenMMRec/src
python run_rf_models.py --dataset <dataset> --models <models> --use_rf <true/false>
```

### 参数说明

- `--dataset`: 数据集名称
  - 可选值：`baby`, `sports`, `clothing`, `all`
  - 必需参数

- `--models`: 要运行的模型
  - 可选值：`RFLGMRec`, `RFBM3`, `RFFREEDOM`, `all`
  - 可以指定多个模型，用空格分隔
  - 必需参数

- `--use_rf`: RF增强开关
  - 可选值：`true`, `false`
  - 可以指定多个值，用空格分隔
  - 默认值：`true false` (两种都测试)

### 使用示例

#### 1. 在单个数据集上测试所有模型（对比 use_rf=true 和 false）

```bash
python run_rf_models.py --dataset baby --models all
```

#### 2. 在所有数据集上测试单个模型

```bash
python run_rf_models.py --dataset all --models RFLGMRec
```

#### 3. 测试特定模型组合

```bash
python run_rf_models.py --dataset sports --models RFLGMRec RFBM3 --use_rf true
```

#### 4. 完整对比实验（所有数据集 × 所有模型 × use_rf 开关）

```bash
python run_rf_models.py --dataset all --models all --use_rf true false
```

这将运行 3 × 3 × 2 = 18 个实验

## 方式二：使用 Bash 脚本（推荐）

### 基本用法

```bash
cd GenMMRec
./run_rf_comparison.sh [选项]
```

### 选项说明

- `-d, --dataset DATASET`: 指定数据集（默认：baby）
- `-m, --models MODELS`: 指定模型（默认：all）
- `-r, --use_rf VALUES`: 指定use_rf值（true/false/both，默认：both）
- `-h, --help`: 显示帮助信息

### 使用示例

#### 1. 使用默认配置（baby数据集，所有模型，对比true和false）

```bash
./run_rf_comparison.sh
```

#### 2. 在所有数据集上测试 RFLGMRec

```bash
./run_rf_comparison.sh -d all -m RFLGMRec -r both
```

#### 3. 在 sports 数据集上测试 RFBM3 和 RFFREEDOM，仅使用 RF 增强

```bash
./run_rf_comparison.sh -d sports -m "RFBM3 RFFREEDOM" -r true
```

#### 4. 完整对比实验

```bash
./run_rf_comparison.sh -d all -m all -r both
```

## 模型特定配置

### RFLGMRec 数据集特定超参数

RFLGMRec 模型会根据不同数据集自动调整以下超参数：

**Baby 数据集：**
- `n_hyper_layer: 1`
- `hyper_num: 4`
- `keep_rate: 0.5`
- `alpha: 0.3`

**Sports 数据集：**
- `n_hyper_layer: 1`
- `hyper_num: 4`
- `keep_rate: 0.4`
- `alpha: 0.6`

**Clothing 数据集：**
- `n_hyper_layer: 2`
- `hyper_num: 64`
- `keep_rate: 0.2`
- `alpha: 0.2`

### RF 增强参数

所有三个模型共享以下 RF 增强参数（在各自的 YAML 配置文件中）：

```yaml
use_rf: True/False
rf_hidden_dim: 128
rf_n_layers: 2
rf_dropout: 0.1
rf_sampling_steps: 10
rf_learning_rate: [0.00005, 0.0001, 0.0003, 0.0005, 0.001]
rf_loss_weight: [0.1, 0.15, 0.2]
rf_warmup_epochs: 5
rf_mix_ratio: 0.1
rf_inference_mix_ratio: 0.2
rf_contrast_temp: 0.2
```

## 配置文件管理

脚本会自动：
1. 备份原始配置文件（.bak）
2. 根据数据集和 use_rf 参数更新配置
3. 运行训练
4. 恢复原始配置

因此，你无需手动修改配置文件。

## 结果汇总

训练完成后，脚本会显示汇总信息：

```
======================================================================
  训练结果汇总
======================================================================
  RFLGMRec-baby-rfTrue: ✓ 成功
  RFLGMRec-baby-rfFalse: ✓ 成功
  RFBM3-baby-rfTrue: ✓ 成功
  ...
======================================================================

  总计: 6 个任务
  成功: 6 个
  失败: 0 个
```

## 注意事项

1. 确保在运行脚本前已经安装了所有依赖项
2. 确保数据集已经下载和预处理完成
3. 批量运行会花费较长时间，建议使用 `tmux` 或 `screen` 在后台运行
4. 每次运行都会自动管理配置文件，无需手动修改
5. 如果某个任务失败，脚本会继续运行后续任务

## 快速开始

最简单的使用方式：

```bash
# 切换到项目目录
cd GenMMRec

# 运行默认配置（baby数据集，所有模型，对比RF开关）
./run_rf_comparison.sh

# 或者运行完整对比实验
./run_rf_comparison.sh -d all -m all -r both
```

## 故障排除

### 配置文件未找到

确保配置文件存在：
- `configs/model/RFLGMRec.yaml`
- `configs/model/RFBM3.yaml`
- `configs/model/RFFREEDOM.yaml`

### 训练失败

检查：
1. 数据集是否已准备好
2. GPU 内存是否足够
3. 查看具体的错误信息

### 权限问题

给脚本添加可执行权限：

```bash
chmod +x run_rf_comparison.sh
chmod +x src/run_rf_models.py
```

#!/bin/bash
# 运行 DiffMM vs RFMRec 对比实验

echo "=============================================="
echo "  DiffMM vs RFMRec 对比实验"
echo "=============================================="
echo ""

# 设置数据集（可修改）
DATASET=${1:-baby}
echo "数据集: $DATASET"
echo ""

# 运行对比脚本
python compare_models.py --models DiffMM RFMREC  --dataset $DATASET

echo ""
echo "=============================================="
echo "对比实验完成！"
echo "=============================================="

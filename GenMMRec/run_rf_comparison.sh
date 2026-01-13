#!/bin/bash
# RF Models 对比测试脚本
# 快速运行所有模型在不同数据集上的对比实验

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 切换到src目录
cd "$(dirname "$0")/src" || exit 1

# 显示帮助信息
show_help() {
    cat << EOF
RF Models 对比测试脚本

用法:
    ./run_rf_comparison.sh [选项]

选项:
    -d, --dataset DATASET    指定数据集 (baby/sports/clothing/all)，默认: baby
    -m, --models MODELS      指定模型 (RFLGMRec/RFBM3/RFFREEDOM/all)，默认: all
    -r, --use_rf VALUES      指定use_rf值 (true/false/both)，默认: both
    -h, --help               显示此帮助信息

示例:
    # 在baby数据集上测试所有模型，对比use_rf=true和false
    ./run_rf_comparison.sh -d baby -m all -r both

    # 在所有数据集上测试RFLGMRec模型，仅use_rf=true
    ./run_rf_comparison.sh -d all -m RFLGMRec -r true

    # 在sports数据集上测试RFBM3和RFFREEDOM，仅use_rf=false
    ./run_rf_comparison.sh -d sports -m "RFBM3 RFFREEDOM" -r false

    # 使用默认设置（baby数据集，所有模型，对比true和false）
    ./run_rf_comparison.sh

EOF
}

# 默认参数
DATASET="all"
MODELS="all"
USE_RF="both"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--dataset)
            DATASET="$2"
            shift 2
            ;;
        -m|--models)
            MODELS="$2"
            shift 2
            ;;
        -r|--use_rf)
            USE_RF="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            print_error "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

# 转换use_rf参数
if [ "$USE_RF" = "both" ]; then
    USE_RF_ARGS="true false"
elif [ "$USE_RF" = "true" ] || [ "$USE_RF" = "false" ]; then
    USE_RF_ARGS="$USE_RF"
else
    print_error "无效的use_rf值: $USE_RF (应为 true/false/both)"
    exit 1
fi

# 显示运行配置
print_info "运行配置:"
echo "  数据集: $DATASET"
echo "  模型: $MODELS"
echo "  use_rf: $USE_RF_ARGS"
echo ""

# 构建命令
if [ "$MODELS" = "all" ]; then
    MODELS_ARG="all"
else
    MODELS_ARG="$MODELS"
fi

# 运行训练脚本
print_info "开始批量训练..."
python run_rf_models.py \
    --dataset "$DATASET" \
    --models $MODELS_ARG \
    --use_rf $USE_RF_ARGS

# 检查退出状态
if [ $? -eq 0 ]; then
    print_info "批量训练完成！"
else
    print_error "批量训练失败！"
    exit 1
fi

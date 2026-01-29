#!/usr/bin/env python3
"""
RF Models 批量训练脚本
用法:
  python run_rf_models.py --dataset baby --models all
  python run_rf_models.py --dataset sports --models RFLGMRec RFBM3
  python run_rf_models.py --dataset all --models all --use_rf true false
  python run_rf_models.py --dataset baby --models RFLGMRec --use_rf true
"""

import argparse
import subprocess
import sys
import os
import yaml
from typing import List, Dict, Any


# 模型-数据集特定配置
MODEL_DATASET_CONFIGS = {
    "RFLGMRec": {
        "baby": {
            "n_hyper_layer": 1,
            "hyper_num": 4,
            "keep_rate": 0.5,
            "alpha": 0.3,
        },
        "sports": {
            "n_hyper_layer": 1,
            "hyper_num": 4,
            "keep_rate": 0.4,
            "alpha": 0.6,
        },
        "clothing": {
            "n_hyper_layer": 2,
            "hyper_num": 64,
            "keep_rate": 0.2,
            "alpha": 0.2,
        },
    },
    "RFBM3": {
        "baby": {
            "embedding_size": 64,
            "feat_embed_dim": 64,
            "n_layers": 1,
            "dropout": 0.3,
            "reg_weight": 0.1,
            "cl_weight": 2.0,
            "use_neg_sampling": False,
        },
        "sports": {
            "embedding_size": 64,
            "feat_embed_dim": 64,
            "n_layers": 1,
            "dropout": 0.5,
            "reg_weight": 0.1,
            "cl_weight": 2.0,
            "use_neg_sampling": False,
        },
        "clothing": {
            "embedding_size": 64,
            "feat_embed_dim": 64,
            "n_layers": 1,
            "dropout": 0.3,
            "reg_weight": 0.1,
            "cl_weight": 2.0,
            "use_neg_sampling": False,
        },
    },
    "RFSMORE": {
        "baby": {
            "n_ui_layers": 4,
            "reg_weight": 1e-4,
            "cl_loss": 0.01,
            "image_knn_k": 40,
            "text_knn_k": 15,
            "dropout_rate": 0.1,
        },
        "sports": {
            "n_ui_layers": 3,
            "reg_weight": 1e-4,
            "cl_loss": 0.03,
            "image_knn_k": 10,
            "text_knn_k": 10,
            "dropout_rate": 0,
        },
        "clothing": {
            "n_ui_layers": 3,
            "reg_weight": 1e-5,
            "cl_loss": 0.01,
            "image_knn_k": 40,
            "text_knn_k": 10,
            "dropout_rate": 0,
        },
        "microlens": {
            "n_ui_layers": 3,
            "reg_weight": 1e-5,
            "cl_loss": 0.01,
            "image_knn_k": 40,
            "text_knn_k": 10,
            "dropout_rate": 0,
        },
    },
    "RFCOHESION": {
        "baby": {
            "reg_weight": 0.0001,
            "num_layer": 1,
        },
        "sports": {
            "reg_weight": 0.001,
            "num_layer": 2,
        },
        "clothing": {
            "reg_weight": 0.001,
            "num_layer": 2,
        },
        "microlens": {
            "reg_weight": 0.001,
            "num_layer": 2,
        },
    },
    "RFDualGNN": {
        "baby": {
            "reg_weight": 0.01,
        },
        "sports": {
            "reg_weight": 0.1,
        },
        "clothing": {
            "reg_weight": 0.1,
        },
        "microlens": {
            "reg_weight": 0.1,
        },
    },
    "RFLATTICE": {
        "baby": {
            "reg_weight": 0.001,
        },
        "sports": {
            "reg_weight": 0.0,
        },
        "clothing": {
            "reg_weight": 0.0,
        },
        "microlens": {
            "reg_weight": 0.0,
        },
    },
    "RFMGCN": {
        "baby": {
            "cl_loss": 0.001,
        },
        "sports": {
            "cl_loss": 0.01,
        },
        "clothing": {
            "cl_loss": 0.01,
        },
        "microlens": {
            "cl_loss": 0.01,
        },
    },
    "RFGUME": {
        "baby": {
            "n_layers": 2,
            "bm_temp": 0.4,
            "um_loss": 0.01,
            "um_temp": 0.1,
            "vt_loss": 0.1,
        },
        "sports": {
            "n_layers": 1,
            "bm_temp": 0.2,
            "um_loss": 0.01,
            "um_temp": 0.1,
            "vt_loss": 0.01,
        },
        "clothing": {
            "n_layers": 1,
            "bm_temp": 0.2,
            "um_loss": 0.1,
            "um_temp": 0.2,
            "vt_loss": 0.001,
        },
    },
}

# 所有支持的模型和配置文件（当前优先搜索的模型）
MODEL_CONFIGS = {
    "RFVBPR": "configs/model/RFVBPR.yaml",
    "RFBM3": "configs/model/RFBM3.yaml",
    "RFFREEDOM": "configs/model/RFFREEDOM.yaml",
    "RFMGCN": "configs/model/RFMGCN.yaml",
    "RFLGMRec": "configs/model/RFLGMRec.yaml",
    "RFSMORE": "configs/model/RFSMORE.yaml",
    "RFGUME": "configs/model/RFGUME.yaml",
    "RFCOHESION": "configs/model/RFCOHESION.yaml",
}

# 其他 RF 模型（暂时不搜索）
# MODEL_CONFIGS_LATER = {
#     "RFMMGCN": "configs/model/RFMMGCN.yaml",
#     "RFLATTICE": "configs/model/RFLATTICE.yaml",
#     "RFDualGNN": "configs/model/RFDualGNN.yaml",
#     "RFPGL": "configs/model/RFPGL.yaml",
# }

# 支持的数据集
DATASETS = ["baby", "sports", "clothing", "microlens"]


def backup_config(config_path: str) -> str:
    """备份配置文件"""
    backup_path = config_path + ".bak"
    with open(config_path, "r") as f:
        original_config = f.read()
    with open(backup_path, "w") as f:
        f.write(original_config)
    return backup_path


def restore_config(config_path: str, backup_path: str):
    """恢复配置文件"""
    if os.path.exists(backup_path):
        with open(backup_path, "r") as f:
            backup_config = f.read()
        with open(config_path, "w") as f:
            f.write(backup_config)
        os.remove(backup_path)


def update_model_config(model: str, dataset: str, config_path: str):
    """更新模型配置文件中的数据集特定参数"""
    if model not in MODEL_DATASET_CONFIGS:
        return  # 该模型没有数据集特定配置

    model_configs = MODEL_DATASET_CONFIGS[model]
    if dataset not in model_configs:
        raise ValueError(
            f"不支持的数据集: {dataset}. 支持: {list(model_configs.keys())}"
        )

    dataset_config = model_configs[dataset]

    # 读取配置
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 更新数据集特定参数
    for key, value in dataset_config.items():
        config[key] = value

    # 写回配置
    with open(config_path, "w") as f:
        yaml.dump(
            config, f, default_flow_style=False, allow_unicode=True, sort_keys=False
        )

    print(f"  [Config] 已更新 {model} 为 {dataset} 数据集配置:")
    for key, value in dataset_config.items():
        print(f"    {key}: {value}")


def update_use_rf(config_path: str, use_rf: bool):
    """更新配置文件中的use_rf参数，当use_rf为False时清空hyper_parameters"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config["use_rf"] = use_rf

    # 当use_rf为False时，清空hyper_parameters（因为RF相关参数不需要调参）
    if not use_rf:
        config["hyper_parameters"] = []
        print(f"  [Config] use_rf = {use_rf}, hyper_parameters = []")
    else:
        print(f"  [Config] use_rf = {use_rf}")

    with open(config_path, "w") as f:
        yaml.dump(
            config, f, default_flow_style=False, allow_unicode=True, sort_keys=False
        )


def run_training(model: str, dataset: str, config_path: str, use_rf: bool) -> bool:
    """运行单个模型训练"""
    print("\n" + "=" * 70)
    print(f"  Model: {model} | Dataset: {dataset} | use_rf: {use_rf}")
    print("=" * 70)

    # 备份配置
    backup_path = backup_config(config_path)
    print(f"  [Backup] 配置已备份到: {backup_path}")

    try:
        # 更新配置
        update_model_config(model, dataset, config_path)
        update_use_rf(config_path, use_rf)

        # 运行训练
        cmd = ["python", "main.py", "--model", model, "--dataset", dataset]
        print(f"\n  [Run] {' '.join(cmd)}\n")

        result = subprocess.run(cmd, check=True)
        print(f"\n  [Done] {model} - {dataset} (use_rf={use_rf}) 训练完成!")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n  [Error] {model} - {dataset} (use_rf={use_rf}) 训练失败: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n  [Interrupted] {model} - {dataset} (use_rf={use_rf}) 训练被中断")
        return False
    finally:
        # 恢复配置
        restore_config(config_path, backup_path)
        print(f"  [Restore] 配置文件已恢复\n")


def main():
    parser = argparse.ArgumentParser(
        description="RF Models 批量训练脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_rf_models.py --dataset baby --models all
  python run_rf_models.py --dataset sports --models RFLGMRec RFBM3
  python run_rf_models.py --dataset all --models all --use_rf true false
  python run_rf_models.py --dataset baby --models RFLGMRec --use_rf true
        """,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help=f"数据集名称: {', '.join(DATASETS + ['all'])}",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        help=f"模型名称: {', '.join(list(MODEL_CONFIGS.keys()) + ['all'])}",
    )
    parser.add_argument(
        "--use_rf",
        type=str,
        nargs="+",
        default=["true", "false"],
        choices=["true", "false"],
        help="use_rf参数值 (默认: true false，即两种都测试)",
    )

    args = parser.parse_args()

    # 解析数据集
    if args.dataset == "all":
        datasets = DATASETS
    elif args.dataset in DATASETS:
        datasets = [args.dataset]
    else:
        print(f"错误: 不支持的数据集 '{args.dataset}'")
        print(f"支持的数据集: {', '.join(DATASETS + ['all'])}")
        sys.exit(1)

    # 解析模型
    if "all" in args.models:
        models = list(MODEL_CONFIGS.keys())
    else:
        models = []
        for model in args.models:
            if model in MODEL_CONFIGS:
                models.append(model)
            else:
                print(f"错误: 不支持的模型 '{model}'")
                print(f"支持的模型: {', '.join(list(MODEL_CONFIGS.keys()) + ['all'])}")
                sys.exit(1)

    # 解析use_rf参数
    use_rf_values = [val.lower() == "true" for val in args.use_rf]

    # 检查配置文件是否存在
    for model, config_path in MODEL_CONFIGS.items():
        if model in models and not os.path.exists(config_path):
            print(f"错误: 配置文件不存在: {config_path}")
            sys.exit(1)

    # 执行训练
    results = {}
    total_tasks = len(datasets) * len(models) * len(use_rf_values)
    current_task = 0

    print("\n" + "=" * 70)
    print(f"  开始批量训练")
    print("=" * 70)
    print(f"  数据集: {', '.join(datasets)}")
    print(f"  模型: {', '.join(models)}")
    print(f"  use_rf: {', '.join(args.use_rf)}")
    print(f"  总任务数: {total_tasks}")
    print("=" * 70)

    for dataset in datasets:
        for model in models:
            for use_rf in use_rf_values:
                current_task += 1
                print(f"\n>>> 进度: [{current_task}/{total_tasks}]")

                config_path = MODEL_CONFIGS[model]
                success = run_training(model, dataset, config_path, use_rf)

                key = f"{model}-{dataset}-rf{use_rf}"
                results[key] = "✓ 成功" if success else "✗ 失败"

    # 打印汇总结果
    print("\n" + "=" * 70)
    print("  训练结果汇总")
    print("=" * 70)
    for key, status in results.items():
        print(f"  {key}: {status}")
    print("=" * 70)

    # 统计成功/失败数量
    success_count = sum(1 for v in results.values() if "成功" in v)
    fail_count = sum(1 for v in results.values() if "失败" in v)
    print(f"\n  总计: {len(results)} 个任务")
    print(f"  成功: {success_count} 个")
    print(f"  失败: {fail_count} 个")
    print()


if __name__ == "__main__":
    main()

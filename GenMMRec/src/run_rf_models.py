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


# RFLGMRec 数据集特定配置
RFLGMREC_DATASET_CONFIGS = {
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
}

# 所有支持的模型和配置文件
MODEL_CONFIGS = {
    "RFLGMRec": "configs/model/RFLGMRec.yaml",
    "RFBM3": "configs/model/RFBM3.yaml",
    "RFFREEDOM": "configs/model/RFFREEDOM.yaml",
}

# 支持的数据集
DATASETS = ["baby", "sports", "clothing"]


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


def update_rflgmrec_config(dataset: str, config_path: str):
    """更新RFLGMRec.yaml中的数据集特定参数"""
    if dataset not in RFLGMREC_DATASET_CONFIGS:
        raise ValueError(
            f"不支持的数据集: {dataset}. 支持: {list(RFLGMREC_DATASET_CONFIGS.keys())}"
        )

    dataset_config = RFLGMREC_DATASET_CONFIGS[dataset]

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

    print(f"  [Config] 已更新 RFLGMRec 为 {dataset} 数据集配置:")
    for key, value in dataset_config.items():
        print(f"    {key}: {value}")


def update_use_rf(config_path: str, use_rf: bool):
    """更新配置文件中的use_rf参数"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config["use_rf"] = use_rf

    with open(config_path, "w") as f:
        yaml.dump(
            config, f, default_flow_style=False, allow_unicode=True, sort_keys=False
        )

    print(f"  [Config] use_rf = {use_rf}")


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
        if model == "RFLGMRec":
            update_rflgmrec_config(dataset, config_path)

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

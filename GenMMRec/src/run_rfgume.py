#!/usr/bin/env python3
"""
RFGUME 训练脚本
用法: python run_rfgume.py --dataset baby
      python run_rfgume.py --dataset sports
      python run_rfgume.py --dataset clothing
      python run_rfgume.py --dataset all  # 运行所有数据集
"""

import argparse
import subprocess
import sys
import os

# 数据集特定配置
DATASET_CONFIGS = {
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
}


def update_config(dataset: str, config_path: str):
    """更新RFGUME.yaml中的数据集特定参数"""
    import yaml

    if dataset not in DATASET_CONFIGS:
        raise ValueError(
            f"不支持的数据集: {dataset}. 支持: {list(DATASET_CONFIGS.keys())}"
        )

    dataset_config = DATASET_CONFIGS[dataset]

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

    print(f"[Config] 已更新 {config_path} 为 {dataset} 数据集配置:")
    for key, value in dataset_config.items():
        print(f"  {key}: {value}")


def run_training(dataset: str, config_path: str):
    """运行RFGUME训练"""
    print("\n" + "=" * 50)
    print(f"  RFGUME Training - Dataset: {dataset}")
    print("=" * 50)

    # 更新配置
    update_config(dataset, config_path)

    # 运行训练
    cmd = ["python", "main.py", "--model", "RFGUME", "--dataset", dataset]
    print(f"\n[Run] {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n[Done] {dataset} 训练完成!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[Error] {dataset} 训练失败: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n[Interrupted] {dataset} 训练被中断")
        return False


def main():
    parser = argparse.ArgumentParser(description="RFGUME 训练脚本")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["baby", "sports", "clothing", "all"],
        help="数据集名称 (baby/sports/clothing/all)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model/RFGUME.yaml",
        help="RFGUME配置文件路径",
    )

    args = parser.parse_args()

    # 检查配置文件是否存在
    if not os.path.exists(args.config):
        print(f"错误: 配置文件不存在: {args.config}")
        sys.exit(1)

    # 备份原配置
    backup_path = args.config + ".bak"
    with open(args.config, "r") as f:
        original_config = f.read()
    with open(backup_path, "w") as f:
        f.write(original_config)
    print(f"[Backup] 原配置已备份到: {backup_path}")

    try:
        if args.dataset == "all":
            # 运行所有数据集
            results = {}
            # for dataset in ["baby", "sports", "clothing"]:
            for dataset in ["clothing", "sports"]:
                success = run_training(dataset, args.config)
                results[dataset] = "✓ 成功" if success else "✗ 失败"

            # 打印汇总
            print("\n" + "=" * 50)
            print("  训练结果汇总")
            print("=" * 50)
            for dataset, status in results.items():
                print(f"  {dataset}: {status}")
        else:
            # 运行单个数据集
            run_training(args.dataset, args.config)

    finally:
        # 恢复原配置
        with open(backup_path, "r") as f:
            backup_config = f.read()
        with open(args.config, "w") as f:
            f.write(backup_config)
        os.remove(backup_path)
        print(f"\n[Restore] 配置文件已恢复")


if __name__ == "__main__":
    main()

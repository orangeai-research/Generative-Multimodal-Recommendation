#!/usr/bin/env python3
"""
RF-GUME 先验知识指导消融实验脚本

测试四种配置组合:
1. baseline: 无指导 (use_user_guidance=False, use_cosine_guidance=False)
2. user_only: 仅用户先验 (use_user_guidance=True, use_cosine_guidance=False)
3. cosine_only: 仅余弦梯度 (use_user_guidance=False, use_cosine_guidance=True)
4. both: 两者都启用 (use_user_guidance=True, use_cosine_guidance=True)

用法:
    python run_ablation_prior.py --dataset baby
    python run_ablation_prior.py --dataset sports --gpu 0
"""

import argparse
import subprocess
import sys
import os
from datetime import datetime
import yaml

# 数据集特定配置（从 run_rfgume.py 复制）
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

# 消融实验配置
ABLATION_CONFIGS = {
    "baseline": {
        "name": "1_baseline_no_guidance",
        "use_user_guidance": False,
        "use_cosine_guidance": False,
        "description": "Baseline: 无任何先验指导",
    },
    "user_only": {
        "name": "2_user_prior_only",
        "use_user_guidance": True,
        "use_cosine_guidance": False,
        "description": "仅用户兴趣先验指导",
    },
    "cosine_only": {
        "name": "3_cosine_gradient_only",
        "use_user_guidance": False,
        "use_cosine_guidance": True,
        "description": "仅余弦相似度梯度指导",
    },
    "both": {
        "name": "4_both_guidances",
        "use_user_guidance": True,
        "use_cosine_guidance": True,
        "description": "两种指导都启用",
    },
}


def update_config(dataset: str, config_path: str, guidance_config: dict):
    """更新RFGUME.yaml配置"""
    if dataset not in DATASET_CONFIGS:
        raise ValueError(
            f"不支持的数据集: {dataset}. 支持: {list(DATASET_CONFIGS.keys())}"
        )

    # 读取配置
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 更新数据集特定参数
    dataset_config = DATASET_CONFIGS[dataset]
    for key, value in dataset_config.items():
        config[key] = value

    # 更新先验指导参数
    config["use_user_guidance"] = guidance_config["use_user_guidance"]
    config["use_cosine_guidance"] = guidance_config["use_cosine_guidance"]

    # 写回配置
    with open(config_path, "w") as f:
        yaml.dump(
            config, f, default_flow_style=False, allow_unicode=True, sort_keys=False
        )

    print(f"[Config] {guidance_config['description']}")
    print(f"  Dataset: {dataset}")
    print(f"  use_user_guidance: {guidance_config['use_user_guidance']}")
    print(f"  use_cosine_guidance: {guidance_config['use_cosine_guidance']}")


def run_experiment(
    dataset: str, config_path: str, exp_key: str, result_dir: str, gpu_id: int = None
):
    """运行单个消融实验"""
    exp_config = ABLATION_CONFIGS[exp_key]
    exp_name = exp_config["name"]

    print("\n" + "=" * 60)
    print(f"  实验: {exp_name}")
    print(f"  {exp_config['description']}")
    print("=" * 60)

    # 更新配置
    update_config(dataset, config_path, exp_config)

    # 准备日志文件
    log_file = os.path.join(result_dir, f"{exp_name}_{dataset}.log")

    # 设置环境变量
    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"  GPU: {gpu_id}")

    # 运行训练
    cmd = ["python", "main.py", "--model", "RFGUME", "--dataset", dataset]
    print(f"\n[Run] {' '.join(cmd)}")
    print(f"[Log] 输出将保存到: {log_file}\n")

    try:
        with open(log_file, "w") as f:
            # 写入实验信息头
            f.write(f"Experiment: {exp_name}\n")
            f.write(f"Description: {exp_config['description']}\n")
            f.write(f"Dataset: {dataset}\n")
            f.write(f"use_user_guidance: {exp_config['use_user_guidance']}\n")
            f.write(f"use_cosine_guidance: {exp_config['use_cosine_guidance']}\n")
            f.write(f"Started at: {datetime.now()}\n")
            f.write("=" * 60 + "\n\n")
            f.flush()

            # 运行训练，输出到文件和终端
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
            )

            # 实时显示输出并写入日志
            for line in process.stdout:
                print(line, end="")
                f.write(line)
                f.flush()

            process.wait()

            # 写入结束信息
            f.write("\n" + "=" * 60 + "\n")
            f.write(f"Finished at: {datetime.now()}\n")
            f.write(f"Exit code: {process.returncode}\n")

        if process.returncode == 0:
            print(f"\n✓ {exp_name} 完成!")
            return True
        else:
            print(f"\n✗ {exp_name} 失败 (exit code: {process.returncode})")
            return False

    except KeyboardInterrupt:
        print(f"\n[Interrupted] {exp_name} 被用户中断")
        return False
    except Exception as e:
        print(f"\n[Error] {exp_name} 出错: {e}")
        return False


def extract_results(result_dir: str, dataset: str):
    """从日志文件中提取结果摘要"""
    summary_file = os.path.join(result_dir, f"results_summary_{dataset}.txt")

    with open(summary_file, "w") as summary:
        summary.write(f"消融实验结果摘要 - Dataset: {dataset}\n")
        summary.write(f"{'=' * 60}\n")
        summary.write(f"完成时间: {datetime.now()}\n\n")

        for exp_key in ["baseline", "user_only", "cosine_only", "both"]:
            exp_config = ABLATION_CONFIGS[exp_key]
            exp_name = exp_config["name"]
            log_file = os.path.join(result_dir, f"{exp_name}_{dataset}.log")

            summary.write(f"\n{'-' * 60}\n")
            summary.write(f"实验: {exp_name}\n")
            summary.write(f"{exp_config['description']}\n")
            summary.write(f"{'-' * 60}\n")

            if os.path.exists(log_file):
                try:
                    with open(log_file, "r") as f:
                        log_content = f.read()

                    # 尝试提取关键指标
                    # 查找包含 "best" 和 "test" 的行
                    lines = log_content.split("\n")
                    found_results = False

                    for i, line in enumerate(lines):
                        if "best" in line.lower() and "test" in line.lower():
                            # 打印该行及后续几行
                            summary.write("\n关键结果:\n")
                            for j in range(i, min(i + 10, len(lines))):
                                if lines[j].strip():
                                    summary.write(f"  {lines[j]}\n")
                            found_results = True
                            break

                    # 查找 Recall 和 NDCG
                    summary.write("\n关键指标:\n")
                    for line in lines:
                        if any(
                            metric in line.lower()
                            for metric in ["recall@", "ndcg@", "hit@"]
                        ):
                            summary.write(f"  {line.strip()}\n")

                    if not found_results:
                        summary.write("  未找到明确的结果标记\n")

                except Exception as e:
                    summary.write(f"  读取日志失败: {e}\n")
            else:
                summary.write("  日志文件不存在\n")

        summary.write(f"\n{'=' * 60}\n")
        summary.write(f"所有日志保存在: {result_dir}\n")

    print(f"\n结果摘要已保存到: {summary_file}")
    print("\n摘要内容:")
    print("=" * 60)
    with open(summary_file, "r") as f:
        print(f.read())


def run_all_datasets(args, experiments_to_run):
    """在所有数据集上运行实验"""
    import time

    all_datasets = ["baby", "sports", "clothing"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 创建总结果目录
    master_result_dir = f"log/ablation_results_all_datasets_{timestamp}"
    os.makedirs(master_result_dir, exist_ok=True)

    print("=" * 60)
    print("  RF-GUME 先验知识指导消融实验 - 所有数据集")
    print("=" * 60)
    print(f"数据集: {', '.join(all_datasets)}")
    print(f"GPU: {args.gpu if args.gpu is not None else '默认'}")
    print(f"每个数据集实验数量: {len(experiments_to_run)}")
    print(f"总实验数量: {len(all_datasets) * len(experiments_to_run)}")
    print(f"结果目录: {master_result_dir}")
    print("=" * 60)

    # 备份原配置
    backup_path = args.config + ".bak"
    with open(args.config, "r") as f:
        original_config = f.read()
    with open(backup_path, "w") as f:
        f.write(original_config)
    print(f"[Backup] 原配置已备份到: {backup_path}")

    all_results = {}

    try:
        for dataset_idx, dataset in enumerate(all_datasets, 1):
            print("\n" + "=" * 60)
            print(
                f"  数据集进度: [{dataset_idx}/{len(all_datasets)}] - {dataset.upper()}"
            )
            print("=" * 60)

            # 为每个数据集创建子目录
            dataset_result_dir = os.path.join(master_result_dir, dataset)
            os.makedirs(dataset_result_dir, exist_ok=True)

            # 保存数据集实验信息
            info_file = os.path.join(dataset_result_dir, "experiment_info.txt")
            with open(info_file, "w") as f:
                f.write(f"RF-GUME 先验知识指导消融实验\n")
                f.write(f"{'=' * 60}\n")
                f.write(f"开始时间: {datetime.now()}\n")
                f.write(f"数据集: {dataset}\n")
                f.write(f"GPU: {args.gpu if args.gpu is not None else '默认'}\n\n")
                f.write(f"实验列表:\n")
                for exp_key in experiments_to_run:
                    exp_config = ABLATION_CONFIGS[exp_key]
                    f.write(f"  - {exp_config['name']}: {exp_config['description']}\n")

            # 运行该数据集的所有实验
            dataset_results = {}
            for i, exp_key in enumerate(experiments_to_run, 1):
                print(
                    f"\n[{dataset.upper()}] 实验进度: [{i}/{len(experiments_to_run)}]"
                )
                success = run_experiment(
                    dataset, args.config, exp_key, dataset_result_dir, args.gpu
                )
                dataset_results[exp_key] = success

                # 短暂休息
                if i < len(experiments_to_run):
                    print("\n等待 10 秒后继续下一个实验...")
                    time.sleep(10)

            all_results[dataset] = dataset_results

            # 提取该数据集的结果摘要
            print(f"\n提取 {dataset} 数据集结果...")
            extract_results(dataset_result_dir, dataset)

            # 数据集之间休息
            if dataset_idx < len(all_datasets):
                print(f"\n等待 20 秒后继续下一个数据集...")
                time.sleep(20)

        # 生成总摘要
        print("\n" + "=" * 60)
        print("  所有数据集实验结果汇总")
        print("=" * 60)

        master_summary_file = os.path.join(
            master_result_dir, "all_datasets_summary.txt"
        )
        with open(master_summary_file, "w") as summary:
            summary.write(f"RF-GUME 先验知识指导消融实验 - 所有数据集汇总\n")
            summary.write(f"{'=' * 60}\n")
            summary.write(f"完成时间: {datetime.now()}\n\n")

            for dataset in all_datasets:
                summary.write(f"\n{'-' * 60}\n")
                summary.write(f"数据集: {dataset.upper()}\n")
                summary.write(f"{'-' * 60}\n")

                for exp_key, success in all_results[dataset].items():
                    exp_name = ABLATION_CONFIGS[exp_key]["name"]
                    status = "✓ 成功" if success else "✗ 失败"
                    summary.write(f"  {exp_name}: {status}\n")
                    print(f"  [{dataset}] {exp_name}: {status}")

        print(f"\n总摘要已保存到: {master_summary_file}")

    except KeyboardInterrupt:
        print("\n\n实验被用户中断")
    finally:
        # 恢复原配置
        with open(backup_path, "r") as f:
            backup_config = f.read()
        with open(args.config, "w") as f:
            f.write(backup_config)
        os.remove(backup_path)
        print(f"\n[Restore] 配置文件已恢复")

    print("\n" + "=" * 60)
    print(f"所有结果保存在: {master_result_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="RF-GUME 先验知识指导消融实验")
    parser.add_argument(
        "--dataset",
        type=str,
        default="baby",
        choices=["baby", "sports", "clothing", "all"],
        help="数据集名称 (baby/sports/clothing/all)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model/RFGUME.yaml",
        help="RFGUME配置文件路径",
    )
    parser.add_argument(
        "--gpu", type=int, default=None, help="指定GPU编号 (不指定则使用默认GPU)"
    )
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        choices=["baseline", "user_only", "cosine_only", "both", "all"],
        default=["all"],
        help="要运行的实验 (可指定多个，或使用 'all' 运行全部)",
    )

    args = parser.parse_args()

    # 检查配置文件是否存在
    if not os.path.exists(args.config):
        print(f"错误: 配置文件不存在: {args.config}")
        sys.exit(1)

    # 确定要运行的实验
    if "all" in args.experiments:
        experiments_to_run = ["baseline", "user_only", "cosine_only", "both"]
    else:
        experiments_to_run = args.experiments

    # 如果选择了所有数据集
    if args.dataset == "all":
        run_all_datasets(args, experiments_to_run)
        return

    # 单个数据集的处理逻辑
    # 创建结果目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = f"log/ablation_results_{args.dataset}_{timestamp}"
    os.makedirs(result_dir, exist_ok=True)

    print("=" * 60)
    print("  RF-GUME 先验知识指导消融实验")
    print("=" * 60)
    print(f"数据集: {args.dataset}")
    print(f"GPU: {args.gpu if args.gpu is not None else '默认'}")
    print(f"实验数量: {len(experiments_to_run)}")
    print(f"结果目录: {result_dir}")
    print("=" * 60)

    # 备份原配置
    backup_path = args.config + ".bak"
    with open(args.config, "r") as f:
        original_config = f.read()
    with open(backup_path, "w") as f:
        f.write(original_config)
    print(f"[Backup] 原配置已备份到: {backup_path}")

    # 保存实验信息
    info_file = os.path.join(result_dir, "experiment_info.txt")
    with open(info_file, "w") as f:
        f.write(f"RF-GUME 先验知识指导消融实验\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"开始时间: {datetime.now()}\n")
        f.write(f"数据集: {args.dataset}\n")
        f.write(f"GPU: {args.gpu if args.gpu is not None else '默认'}\n\n")
        f.write(f"实验列表:\n")
        for exp_key in experiments_to_run:
            exp_config = ABLATION_CONFIGS[exp_key]
            f.write(f"  - {exp_config['name']}: {exp_config['description']}\n")

    try:
        # 运行实验
        results = {}
        for i, exp_key in enumerate(experiments_to_run, 1):
            print(f"\n进度: [{i}/{len(experiments_to_run)}]")
            success = run_experiment(
                args.dataset, args.config, exp_key, result_dir, args.gpu
            )
            results[exp_key] = success

            # 短暂休息
            if i < len(experiments_to_run):
                print("\n等待 10 秒后继续下一个实验...")
                import time

                time.sleep(10)

        # 打印结果汇总
        print("\n" + "=" * 60)
        print("  实验结果汇总")
        print("=" * 60)
        for exp_key, success in results.items():
            exp_name = ABLATION_CONFIGS[exp_key]["name"]
            status = "✓ 成功" if success else "✗ 失败"
            print(f"  {exp_name}: {status}")

        # 提取结果摘要
        print("\n提取结果...")
        extract_results(result_dir, args.dataset)

    except KeyboardInterrupt:
        print("\n\n实验被用户中断")
    finally:
        # 恢复原配置
        with open(backup_path, "r") as f:
            backup_config = f.read()
        with open(args.config, "w") as f:
            f.write(backup_config)
        os.remove(backup_path)
        print(f"\n[Restore] 配置文件已恢复")

    print("\n" + "=" * 60)
    print(f"所有结果保存在: {result_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
RF Models 分阶段超参数搜索脚本

分阶段搜索策略：
  阶段1: 搜索 rf_loss_weight (3个值)
  阶段2: 固定最优 rf_loss_weight，搜索 rf_learning_rate (3个值)
  阶段3: 固定前两个，搜索 rf_inference_mix_ratio (3个值)
  
总共: 3 + 3 + 3 = 9 次实验，而不是 3x3x3 = 27 次

用法:
  # 运行阶段1（搜索 rf_loss_weight）
  python run_rf_staged_search.py --dataset baby --models all --stage 1
  
  # 运行阶段2（搜索 rf_learning_rate，使用阶段1的最优值）
  python run_rf_staged_search.py --dataset baby --models all --stage 2
  
  # 运行阶段3（搜索 rf_inference_mix_ratio，使用阶段1和2的最优值）
  python run_rf_staged_search.py --dataset baby --models all --stage 3
  
  # 一次性运行所有阶段（自动化）
  python run_rf_staged_search.py --dataset baby --models all --stage all
"""

import argparse
import subprocess
import sys
import os
import yaml
import json
from typing import List, Dict, Any
from pathlib import Path


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

# 支持的数据集
DATASETS = ["baby", "sports", "clothing", "microlens"]

# 分阶段搜索配置
STAGE_CONFIGS = {
    1: {
        "param": "rf_loss_weight",
        "search_values": [0.2, 0.4, 0.6],
        "fixed_params": {
            "rf_learning_rate": 0.0003,
            "rf_inference_mix_ratio": 0.05,
        },
        "hyper_parameters": ["rf_loss_weight"]
    },
    2: {
        "param": "rf_learning_rate",
        "search_values": [0.0001, 0.0003, 0.0005],
        "fixed_params": {
            "rf_inference_mix_ratio": 0.05,
            # rf_loss_weight will be loaded from stage 1 results
        },
        "hyper_parameters": ["rf_learning_rate"]
    },
    3: {
        "param": "rf_inference_mix_ratio",
        "search_values": [0.02, 0.05, 0.1],
        "fixed_params": {
            # rf_loss_weight and rf_learning_rate will be loaded from previous stages
        },
        "hyper_parameters": ["rf_inference_mix_ratio"],
    },
}


def get_results_dir():
    """获取结果保存目录"""
    results_dir = Path("hyperparameter_search_results")
    results_dir.mkdir(exist_ok=True)
    return results_dir


def save_best_params(model: str, dataset: str, stage: int, best_params: Dict[str, Any]):
    """保存最优超参数"""
    results_dir = get_results_dir()
    result_file = results_dir / f"{model}_{dataset}_stage{stage}_best.json"
    
    with open(result_file, "w") as f:
        json.dump(best_params, f, indent=2)
    
    print(f"\n[Save] 最优参数已保存到: {result_file}")
    print(f"       {best_params}")


def load_best_params(model: str, dataset: str, stage: int) -> Dict[str, Any]:
    """加载之前阶段的最优超参数"""
    results_dir = get_results_dir()
    result_file = results_dir / f"{model}_{dataset}_stage{stage}_best.json"
    
    if not result_file.exists():
        print(f"\n[Warning] 未找到阶段{stage}的最优参数文件: {result_file}")
        return {}
    
    with open(result_file, "r") as f:
        best_params = json.load(f)
    
    print(f"\n[Load] 从阶段{stage}加载最优参数: {best_params}")
    return best_params


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


def update_config_for_stage(
    model: str,
    dataset: str,
    config_path: str,
    stage: int,
    previous_best_params: Dict[str, Any] = None,
):
    """更新配置文件为指定阶段的搜索配置"""
    stage_config = STAGE_CONFIGS[stage]
    
    # 读取配置
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # 更新数据集特定参数
    if model in MODEL_DATASET_CONFIGS and dataset in MODEL_DATASET_CONFIGS[model]:
        dataset_config = MODEL_DATASET_CONFIGS[model][dataset]
        for key, value in dataset_config.items():
            config[key] = value
        print(f"  [Config] 已更新 {model} 为 {dataset} 数据集配置")
    
    # 设置超参数搜索列表
    config["hyper_parameters"] = stage_config["hyper_parameters"]
    
    # 设置当前阶段要搜索的参数
    param_name = stage_config["param"]
    config[param_name] = stage_config["search_values"]
    
    # 设置固定参数
    for key, value in stage_config["fixed_params"].items():
        config[key] = value
    
    # 加载之前阶段的最优参数
    if previous_best_params:
        for key, value in previous_best_params.items():
            if key.startswith("rf_"):
                config[key] = value
                print(f"  [Config] 使用之前阶段的最优参数: {key} = {value}")
    
    # 更新 wandb project
    config["wandb_project"] = stage_config["wandb_project"]
    
    # 写回配置
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    print(f"  [Config] 阶段{stage}: 搜索 {param_name} = {stage_config['search_values']}")


def parse_log_results(model: str, dataset: str, stage: int) -> Dict[str, Any]:
    """
    从日志文件中解析最优超参数
    日志格式: Parameters: ['seed', 'rf_loss_weight']=(2024, 0.4)
    """
    # 查找最新的日志文件
    log_dir = Path("log")
    if not log_dir.exists():
        print(f"\n[Error] 日志目录不存在: {log_dir}")
        return manual_input_params(stage)
    
    # 查找匹配的日志文件 (格式: {model}-{dataset}-{timestamp}.log)
    log_files = list(log_dir.glob(f"{model}-{dataset}-*.log"))
    
    if not log_files:
        print(f"\n[Warning] 未找到日志文件: {model}-{dataset}-*.log")
        return manual_input_params(stage)
    
    # 获取最新的日志文件
    latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
    print(f"\n[Parse] 解析日志文件: {latest_log}")
    
    try:
        with open(latest_log, "r", encoding="utf-8") as f:
            log_content = f.read()
        
        # 查找 "█████████████ BEST ████████████████" 部分
        best_section_start = log_content.rfind("█████████████ BEST ████████████████")
        
        if best_section_start == -1:
            print(f"\n[Warning] 日志中未找到 BEST 部分")
            return manual_input_params(stage)
        
        # 提取 BEST 部分的内容
        best_section = log_content[best_section_start:best_section_start + 500]
        
        # 解析参数行: Parameters: ['seed', 'rf_loss_weight']=(2024, 0.4)
        import re
        param_pattern = r"Parameters:\s*\[([^\]]+)\]=\(([^)]+)\)"
        match = re.search(param_pattern, best_section)
        
        if not match:
            print(f"\n[Warning] 无法解析参数格式")
            return manual_input_params(stage)
        
        # 提取参数名和值
        param_names_str = match.group(1)
        param_values_str = match.group(2)
        
        # 清理并分割
        param_names = [name.strip().strip("'\"") for name in param_names_str.split(",")]
        param_values_raw = [val.strip() for val in param_values_str.split(",")]
        
        # 转换值的类型
        param_values = []
        for val in param_values_raw:
            try:
                # 尝试转换为数字
                if "." in val:
                    param_values.append(float(val))
                else:
                    param_values.append(int(val))
            except ValueError:
                param_values.append(val)
        
        # 构建参数字典
        params_dict = dict(zip(param_names, param_values))
        
        # 移除 seed 参数
        if "seed" in params_dict:
            del params_dict["seed"]
        
        print(f"\n[Success] 从日志解析到最优参数: {params_dict}")
        
        # 验证是否包含当前阶段的参数
        stage_param = STAGE_CONFIGS[stage]["param"]
        if stage_param not in params_dict:
            print(f"\n[Warning] 日志中未找到阶段{stage}的参数 '{stage_param}'")
            return manual_input_params(stage)
        
        return params_dict
        
    except Exception as e:
        print(f"\n[Error] 解析日志文件失败: {e}")
        import traceback
        traceback.print_exc()
        return manual_input_params(stage)


def manual_input_params(stage: int) -> Dict[str, Any]:
    """手动输入参数（作为备选方案）"""
    print(f"\n{'='*70}")
    print(f"  请手动输入阶段{stage}的最优超参数")
    print(f"{'='*70}")
    
    stage_param = STAGE_CONFIGS[stage]["param"]
    param_value = input(f"  {stage_param} 的最优值: ").strip()
    
    # 尝试转换为数字
    try:
        if "." in param_value:
            param_value = float(param_value)
        else:
            param_value = int(param_value)
    except ValueError:
        pass
    
    return {stage_param: param_value}


def run_stage(
    model: str,
    dataset: str,
    config_path: str,
    stage: int,
    previous_best_params: Dict[str, Any] = None,
) -> bool:
    """运行单个阶段的搜索"""
    print(f"\n{'='*70}")
    print(f"  阶段{stage}: {model} - {dataset}")
    print(f"  搜索参数: {STAGE_CONFIGS[stage]['param']}")
    print(f"{'='*70}")
    
    # 备份配置
    backup_path = backup_config(config_path)
    
    try:
        # 更新配置为当前阶段
        update_config_for_stage(model, dataset, config_path, stage, previous_best_params)
        
        # 运行训练
        cmd = ["python", "main.py", "--model", model, "--dataset", dataset]
        print(f"\n  [Run] {' '.join(cmd)}\n")
        
        result = subprocess.run(cmd, check=True)
        print(f"\n  [Done] 阶段{stage} 完成!")
        
        # 从日志文件解析最优参数
        best_params = parse_log_results(model, dataset, stage)
        save_best_params(model, dataset, stage, best_params)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n  [Error] 阶段{stage} 失败: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n  [Interrupted] 阶段{stage} 被中断")
        return False
    finally:
        # 恢复配置
        restore_config(config_path, backup_path)


def run_all_stages(model: str, dataset: str, config_path: str):
    """运行所有阶段的搜索"""
    print(f"\n{'='*70}")
    print(f"  开始分阶段搜索: {model} - {dataset}")
    print(f"  总共3个阶段，每个阶段3次实验，共9次实验")
    print(f"{'='*70}")
    
    previous_best_params = {}
    
    for stage in [1, 2, 3]:
        success = run_stage(model, dataset, config_path, stage, previous_best_params)
        
        if not success:
            print(f"\n[Error] 阶段{stage}失败，停止后续阶段")
            return False
        
        # 加载当前阶段的最优参数，用于下一阶段
        stage_best = load_best_params(model, dataset, stage)
        previous_best_params.update(stage_best)
    
    print(f"\n{'='*70}")
    print(f"  {model} - {dataset} 所有阶段完成!")
    print(f"  最终最优参数: {previous_best_params}")
    print(f"{'='*70}")
    
    # 保存最终最优参数
    results_dir = get_results_dir()
    final_result_file = results_dir / f"{model}_{dataset}_final_best.json"
    with open(final_result_file, "w") as f:
        json.dump(previous_best_params, f, indent=2)
    print(f"\n[Save] 最终最优参数已保存到: {final_result_file}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="RF Models 分阶段超参数搜索脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help=f"数据集名称: {', '.join(DATASETS)}",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        help=f"模型名称: {', '.join(list(MODEL_CONFIGS.keys()) + ['all'])}",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["1", "2", "3", "all"],
        help="搜索阶段: 1=rf_loss_weight, 2=rf_learning_rate, 3=rf_inference_mix_ratio, all=所有阶段",
    )
    
    args = parser.parse_args()
    
    # 解析数据集
    if args.dataset not in DATASETS:
        print(f"错误: 不支持的数据集 '{args.dataset}'")
        print(f"支持的数据集: {', '.join(DATASETS)}")
        sys.exit(1)
    
    dataset = args.dataset
    
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
    
    # 检查配置文件是否存在
    for model in models:
        config_path = MODEL_CONFIGS[model]
        if not os.path.exists(config_path):
            print(f"错误: 配置文件不存在: {config_path}")
            sys.exit(1)
    
    # 执行搜索
    print(f"\n{'='*70}")
    print(f"  分阶段超参数搜索")
    print(f"{'='*70}")
    print(f"  数据集: {dataset}")
    print(f"  模型: {', '.join(models)}")
    print(f"  阶段: {args.stage}")
    print(f"{'='*70}")
    
    results = {}
    
    for model in models:
        config_path = MODEL_CONFIGS[model]
        
        if args.stage == "all":
            # 运行所有阶段
            success = run_all_stages(model, dataset, config_path)
            results[f"{model}-{dataset}"] = "✓ 成功" if success else "✗ 失败"
        else:
            # 运行单个阶段
            stage = int(args.stage)
            
            # 加载之前阶段的最优参数
            previous_best_params = {}
            for prev_stage in range(1, stage):
                stage_best = load_best_params(model, dataset, prev_stage)
                previous_best_params.update(stage_best)
            
            success = run_stage(model, dataset, config_path, stage, previous_best_params)
            results[f"{model}-{dataset}-stage{stage}"] = "✓ 成功" if success else "✗ 失败"
    
    # 打印汇总结果
    print(f"\n{'='*70}")
    print(f"  搜索结果汇总")
    print(f"{'='*70}")
    for key, status in results.items():
        print(f"  {key}: {status}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

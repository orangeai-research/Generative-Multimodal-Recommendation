#!/usr/bin/env python
# coding: utf-8
"""
对比实验脚本：DiffMM vs RFMRec
运行两个模型并对比结果
"""

import os
import sys
import subprocess
import json
import re
from datetime import datetime
import argparse

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def run_model(model_name, dataset='baby'):
    """运行指定模型并捕获结果"""
    print(f"\n{'='*60}")
    print(f"开始训练 {model_name} 模型...")
    print(f"{'='*60}\n")

    # 设置工作目录为 src（因为 configurator 从 getcwd 找配置文件）
    base_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(base_dir, 'src')

    # 运行命令
    cmd = [
        'python', 'main.py',
        '-m', model_name,
        '-d', dataset
    ]

    try:
        # 运行并捕获输出（工作目录设为 src）
        result = subprocess.run(
            cmd,
            cwd=src_dir,
            capture_output=True,
            text=True,
            timeout=7200  # 2小时超时
        )

        output = result.stdout + result.stderr

        # 保存完整日志
        log_dir = 'comparison_logs'
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'{model_name}_{dataset}_{timestamp}.log')

        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(output)

        print(f"✅ {model_name.upper()} 训练完成！日志保存在: {log_file}")

        # 解析结果
        metrics = parse_results(output)
        return metrics, log_file

    except subprocess.TimeoutExpired:
        print(f"❌ {model_name.upper()} 训练超时（2小时）")
        return None, None
    except Exception as e:
        print(f"❌ {model_name.upper()} 训练出错: {str(e)}")
        return None, None


def parse_results(output):
    """从日志输出中解析指标"""
    metrics = {
        'valid': {},
        'test': {}
    }

    # 查找最佳结果部分
    best_section = re.search(r'████Current BEST████:(.*?)(?=\n\n|$)', output, re.DOTALL)
    if not best_section:
        best_section = re.search(r'█████████████ BEST ████████████████(.*?)(?=\n\n|$)', output, re.DOTALL)

    if best_section:
        text = best_section.group(1)

        # 解析 Valid 结果
        valid_match = re.search(r'Valid:\s*\{([^}]+)\}', text)
        if valid_match:
            valid_str = valid_match.group(1)
            for item in valid_str.split(','):
                item = item.strip()
                if ':' in item:
                    key, value = item.split(':', 1)
                    key = key.strip().strip("'\"")
                    value = value.strip()
                    try:
                        metrics['valid'][key] = float(value)
                    except ValueError:
                        pass

        # 解析 Test 结果
        test_match = re.search(r'Test:\s*\{([^}]+)\}', text)
        if test_match:
            test_str = test_match.group(1)
            for item in test_str.split(','):
                item = item.strip()
                if ':' in item:
                    key, value = item.split(':', 1)
                    key = key.strip().strip("'\"")
                    value = value.strip()
                    try:
                        metrics['test'][key] = float(value)
                    except ValueError:
                        pass

    return metrics


def compare_results(results):
    """对比并展示结果"""
    print(f"\n{'='*80}")
    print(f"{'模型对比结果':^80}")
    print(f"{'='*80}\n")

    models = list(results.keys())
    if len(models) < 2:
        print("⚠️  只有一个模型的结果，无法对比")
        return

    # 获取所有指标
    all_metrics = set()
    for model_results in results.values():
        if model_results:
            all_metrics.update(model_results['valid'].keys())
            all_metrics.update(model_results['test'].keys())

    # 分别展示 Validation 和 Test 结果
    for split in ['valid', 'test']:
        print(f"\n{'─'*80}")
        print(f"{split.upper()} SET 结果对比")
        print(f"{'─'*80}")

        # 表头
        header = f"{'Metric':<20}"
        for model in models:
            header += f"{model.upper():>15}"
        header += f"{'Winner':>15}"
        print(header)
        print('─' * 80)

        # 每个指标
        metrics_list = sorted([m for m in all_metrics if '@' in m])  # 只显示带@的指标

        for metric in metrics_list:
            row = f"{metric:<20}"
            values = []

            for model in models:
                if results[model] and metric in results[model][split]:
                    value = results[model][split][metric]
                    values.append((model, value))
                    row += f"{value:>15.4f}"
                else:
                    values.append((model, 0))
                    row += f"{'N/A':>15}"

            # 找出最优
            if values and any(v[1] > 0 for v in values):
                best_model = max(values, key=lambda x: x[1])[0]
                row += f"{best_model.upper():>15}"
            else:
                row += f"{'N/A':>15}"

            print(row)

    # 统计胜率
    print(f"\n{'─'*80}")
    print(f"胜率统计 (在 TEST SET 上)")
    print(f"{'─'*80}")

    win_counts = {model: 0 for model in models}
    total_metrics = 0

    metrics_list = sorted([m for m in all_metrics if '@' in m])
    for metric in metrics_list:
        values = []
        for model in models:
            if results[model] and metric in results[model]['test']:
                value = results[model]['test'][metric]
                values.append((model, value))

        if values and any(v[1] > 0 for v in values):
            best_model = max(values, key=lambda x: x[1])[0]
            win_counts[best_model] += 1
            total_metrics += 1

    for model in models:
        win_rate = (win_counts[model] / total_metrics * 100) if total_metrics > 0 else 0
        print(f"{model.upper():<15} 胜出 {win_counts[model]}/{total_metrics} 指标 ({win_rate:.1f}%)")

    # 重点指标对比
    print(f"\n{'─'*80}")
    print(f"重点指标对比 (Test Set)")
    print(f"{'─'*80}")

    key_metrics = ['recall@20', 'ndcg@20', 'precision@20', 'map@20']

    for metric in key_metrics:
        row = f"{metric.upper():<20}"
        values = []

        for model in models:
            if results[model] and metric in results[model]['test']:
                value = results[model]['test'][metric]
                values.append((model, value))
                row += f"{model.upper()}: {value:.4f}  "

        if len(values) == 2:
            improvement = ((values[1][1] - values[0][1]) / values[0][1] * 100) if values[0][1] > 0 else 0
            row += f"  (改进: {improvement:+.2f}%)"

        print(row)

    print(f"\n{'='*80}\n")


def save_comparison_report(results, output_file='comparison_results.json'):
    """保存对比结果为 JSON"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    report = {
        'timestamp': timestamp,
        'models': results
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"📊 对比报告已保存: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='对比 DiffMM 和 RFMRec 模型')
    parser.add_argument('--models', type=str, nargs='+', default=['DiffMM', 'RFMREC'],
                        help='要对比的模型列表 (默认: DiffMM RFMREC)')
    parser.add_argument('--dataset', type=str, default='baby',
                        help='数据集名称 (默认: baby)')
    parser.add_argument('--output', type=str, default='comparison_results.json',
                        help='输出文件名 (默认: comparison_results.json)')

    args = parser.parse_args()

    print(f"\n{'#'*80}")
    print(f"{'模型对比实验':^80}")
    print(f"{'#'*80}")
    print(f"\n模型: {', '.join([m for m in args.models])}")
    print(f"数据集: {args.dataset}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    results = {}

    # 依次运行每个模型
    for model in args.models:
        metrics, log_file = run_model(model, args.dataset)
        results[model] = metrics

        if metrics:
            print(f"\n📋 {model} 关键指标预览:")
            if 'test' in metrics and 'recall@20' in metrics['test']:
                print(f"   Recall@20:  {metrics['test']['recall@20']:.4f}")
                print(f"   NDCG@20:    {metrics['test']['ndcg@20']:.4f}")

        print()

    # 对比结果
    compare_results(results)

    # 保存报告
    save_comparison_report(results, args.output)

    print(f"\n✅ 所有实验完成！")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


if __name__ == '__main__':
    main()

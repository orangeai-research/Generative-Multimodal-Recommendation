#!/usr/bin/env python
# coding: utf-8
"""
RF Enhancement Results Comparison Script

This script parses experimental results from log files and generates
a comprehensive comparison table between original and RF-enhanced models.
"""

import argparse
import os
import re
import pandas as pd
from typing import Dict, List, Tuple


def parse_log_file(log_file: str) -> Dict[str, float]:
    """
    Parse metrics from a log file.

    Args:
        log_file: Path to the log file

    Returns:
        Dictionary of metrics with their best values
    """
    metrics = {}

    if not os.path.exists(log_file):
        print(f"Warning: Log file not found: {log_file}")
        return metrics

    try:
        with open(log_file, 'r') as f:
            content = f.read()

            # Extract best epoch results
            # Look for patterns like "recall@20 : 0.0234"
            patterns = {
                'recall@5': r'recall@5\s*:\s*([\d.]+)',
                'recall@10': r'recall@10\s*:\s*([\d.]+)',
                'recall@20': r'recall@20\s*:\s*([\d.]+)',
                'recall@50': r'recall@50\s*:\s*([\d.]+)',
                'ndcg@5': r'ndcg@5\s*:\s*([\d.]+)',
                'ndcg@10': r'ndcg@10\s*:\s*([\d.]+)',
                'ndcg@20': r'ndcg@20\s*:\s*([\d.]+)',
                'precision@5': r'precision@5\s*:\s*([\d.]+)',
                'precision@10': r'precision@10\s*:\s*([\d.]+)',
            }

            for metric_name, pattern in patterns.items():
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    # Take the best value (usually the last one or max)
                    values = [float(m) for m in matches]
                    metrics[metric_name] = max(values)

    except Exception as e:
        print(f"Error parsing {log_file}: {e}")

    return metrics


def compare_results(datasets: List[str], models: List[str], log_dir: str = "log") -> pd.DataFrame:
    """
    Compare results between original and RF-enhanced models.

    Args:
        datasets: List of dataset names
        models: List of model names (without RF prefix)
        log_dir: Directory containing log files

    Returns:
        DataFrame with comparison results
    """
    results = []

    for dataset in datasets:
        for model in models:
            # Parse original model results
            ori_log = os.path.join(log_dir, f"{model}-{dataset}.txt")
            ori_metrics = parse_log_file(ori_log)

            # Parse RF-enhanced model results
            rf_log = os.path.join(log_dir, f"RF{model}-{dataset}.txt")
            rf_metrics = parse_log_file(rf_log)

            if not ori_metrics or not rf_metrics:
                print(f"Skipping {model} on {dataset} - missing results")
                continue

            # Compare key metrics
            row = {
                'Dataset': dataset,
                'Model': model,
            }

            # Add metrics
            for metric in ['recall@20', 'ndcg@10', 'precision@5']:
                if metric in ori_metrics and metric in rf_metrics:
                    ori_val = ori_metrics[metric]
                    rf_val = rf_metrics[metric]
                    improvement = ((rf_val - ori_val) / ori_val * 100) if ori_val > 0 else 0

                    row[f'Ori_{metric}'] = f"{ori_val:.4f}"
                    row[f'RF_{metric}'] = f"{rf_val:.4f}"
                    row[f'Δ_{metric}(%)'] = f"{improvement:+.2f}"

            results.append(row)

    # Create DataFrame
    df = pd.DataFrame(results)

    return df


def generate_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a summary table with average improvements per model.

    Args:
        df: Comparison DataFrame

    Returns:
        Summary DataFrame
    """
    summary = []

    for model in df['Model'].unique():
        model_df = df[df['Model'] == model]

        # Calculate average improvements
        row = {'Model': model}

        for metric in ['recall@20', 'ndcg@10', 'precision@5']:
            col_name = f'Δ_{metric}(%)'
            if col_name in model_df.columns:
                values = model_df[col_name].str.replace('+', '').astype(float)
                row[f'Avg_Δ_{metric}(%)'] = f"{values.mean():.2f}"

        summary.append(row)

    return pd.DataFrame(summary)


def main():
    parser = argparse.ArgumentParser(description='Compare RF enhancement results')
    parser.add_argument('--datasets', nargs='+', default=['baby', 'clothing', 'sports'],
                        help='List of datasets to compare')
    parser.add_argument('--models', nargs='+', default=['FREEDOM', 'BM3', 'LGMRec'],
                        help='List of models to compare')
    parser.add_argument('--log_dir', type='str', default='log',
                        help='Directory containing log files')
    parser.add_argument('--output', type='str', default='rf_comparison_results.csv',
                        help='Output CSV file path')

    args = parser.parse_args()

    print("=" * 60)
    print("RF Enhancement Results Comparison")
    print("=" * 60)
    print()

    # Compare results
    print("Parsing log files and comparing results...")
    df = compare_results(args.datasets, args.models, args.log_dir)

    if df.empty:
        print("No results found. Please check your log directory and file names.")
        return

    # Display full comparison table
    print("\n" + "=" * 60)
    print("Detailed Comparison Table")
    print("=" * 60)
    print(df.to_string(index=False))

    # Generate and display summary
    print("\n" + "=" * 60)
    print("Summary: Average Improvements per Model")
    print("=" * 60)
    summary_df = generate_summary_table(df)
    print(summary_df.to_string(index=False))

    # Save to CSV
    df.to_csv(args.output, index=False)
    summary_df.to_csv(args.output.replace('.csv', '_summary.csv'), index=False)

    print("\n" + "=" * 60)
    print(f"Results saved to: {args.output}")
    print(f"Summary saved to: {args.output.replace('.csv', '_summary.csv')}")
    print("=" * 60)


if __name__ == "__main__":
    main()

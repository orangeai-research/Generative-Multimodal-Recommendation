#!/usr/bin/env python
# coding: utf-8
"""
Quick test script to verify RF-enhanced models can be instantiated.
This is a basic sanity check before running full experiments.
"""

import sys
import torch

# Add src to path
sys.path.insert(0, 'src')

from utils.utils import get_model


def test_model_loading():
    """Test that all RF-enhanced models can be loaded."""
    models = ['RFFREEDOM', 'RFBM3', 'RFLGMRec']

    print("=" * 60)
    print("Testing RF-Enhanced Model Loading")
    print("=" * 60)
    print()

    results = []

    for model_name in models:
        try:
            model_class = get_model(model_name)
            print(f"✓ {model_name:12s} - Loaded successfully")
            print(f"  Class: {model_class.__name__}")
            print(f"  Module: {model_class.__module__}")
            results.append((model_name, True, None))
        except Exception as e:
            print(f"✗ {model_name:12s} - Failed to load")
            print(f"  Error: {e}")
            results.append((model_name, False, str(e)))
        print()

    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)

    success_count = sum(1 for _, success, _ in results if success)
    total_count = len(results)

    print(f"Successfully loaded: {success_count}/{total_count} models")

    if success_count == total_count:
        print("\n✓ All RF-enhanced models are ready for experiments!")
        return 0
    else:
        print("\n✗ Some models failed to load. Please check the errors above.")
        return 1


def test_rf_parameters():
    """Test that RF parameters are correctly configured."""
    print("\n" + "=" * 60)
    print("Testing RF Parameter Configuration")
    print("=" * 60)
    print()

    # Test loading a config
    import yaml

    config_files = {
        'RFFREEDOM': 'src/configs/model/RFFREEDOM.yaml',
        'RFBM3': 'src/configs/model/RFBM3.yaml',
        'RFLGMRec': 'src/configs/model/RFLGMRec.yaml',
    }

    for model_name, config_path in config_files.items():
        print(f"{model_name}:")
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            rf_params = [k for k in config.keys() if k.startswith('rf_') or k == 'use_rf']
            print(f"  RF Parameters ({len(rf_params)}):")
            for param in sorted(rf_params):
                print(f"    {param:25s} = {config[param]}")
            print(f"  ✓ Configuration loaded successfully")
        except Exception as e:
            print(f"  ✗ Failed to load config: {e}")
        print()


if __name__ == "__main__":
    # Test model loading
    exit_code = test_model_loading()

    # Test parameter configuration
    test_rf_parameters()

    sys.exit(exit_code)

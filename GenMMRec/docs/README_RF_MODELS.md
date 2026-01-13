# RF-Enhanced Models Quick Start Guide

This guide provides quick instructions for using the RF-enhanced versions of FREEDOM, BM3, and LGMRec.

## Quick Verification

Before running experiments, verify that all models are properly installed:

```bash
python test_rf_models.py
```

You should see:
```
✓ All RF-enhanced models are ready for experiments!
```

## Running Individual Experiments

### Basic Usage

```bash
cd src

# Run RF-enhanced FREEDOM on Baby dataset
python main.py --model RFFREEDOM --dataset baby

# Run RF-enhanced BM3 on Baby dataset
python main.py --model RFBM3 --dataset baby

# Run RF-enhanced LGMRec on Baby dataset
python main.py --model RFLGMRec --dataset baby
```

### All Datasets

```bash
# Baby dataset
python main.py --model RFFREEDOM --dataset baby
python main.py --model RFBM3 --dataset baby
python main.py --model RFLGMRec --dataset baby

# Clothing dataset
python main.py --model RFFREEDOM --dataset clothing
python main.py --model RFBM3 --dataset clothing
python main.py --model RFLGMRec --dataset clothing

# Sports dataset
python main.py --model RFFREEDOM --dataset sports
python main.py --model RFBM3 --dataset sports
python main.py --model RFLGMRec --dataset sports
```

## Running Comparison Experiments

To compare original models vs RF-enhanced versions:

```bash
# Run all experiments (original + RF-enhanced)
./run_rf_comparison.sh

# This will run:
# - FREEDOM vs RFFREEDOM
# - BM3 vs RFBM3
# - LGMRec vs RFLGMRec
# on all three datasets (baby, clothing, sports)
```

## Analyzing Results

After experiments complete, analyze the results:

```bash
# Default: compare all models on all datasets
python compare_results.py

# Custom: specify models and datasets
python compare_results.py --models FREEDOM BM3 --datasets baby clothing

# Custom output file
python compare_results.py --output my_results.csv
```

This will generate:
- `rf_comparison_results.csv` - Detailed comparison table
- `rf_comparison_results_summary.csv` - Summary with average improvements

## Model Files

### Implementations
- `src/models/rffreedom.py` - RF-enhanced FREEDOM
- `src/models/rfbm3.py` - RF-enhanced BM3
- `src/models/rflgmrec.py` - RF-enhanced LGMRec

### Configurations
- `src/configs/model/RFFREEDOM.yaml` - RFFREEDOM hyperparameters
- `src/configs/model/RFBM3.yaml` - RFBM3 hyperparameters
- `src/configs/model/RFLGMRec.yaml` - RFLGMRec hyperparameters

## RF Hyperparameters

All RF-enhanced models share these parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_rf` | True | Enable/disable RF enhancement |
| `rf_hidden_dim` | 128 | Hidden dimension of RF network |
| `rf_n_layers` | 2 | Number of residual blocks |
| `rf_dropout` | 0.1 | Dropout rate |
| `rf_learning_rate` | 0.0001 | Independent learning rate for RF |
| `rf_sampling_steps` | 10 | ODE sampling steps |
| `rf_warmup_epochs` | 5 | Epochs before RF activation |
| `rf_mix_ratio` | 0.1 | Training mixing ratio |
| `rf_inference_mix_ratio` | 0.2 | Inference mixing ratio |
| `rf_contrast_temp` | 0.2 | Contrastive loss temperature |
| `rf_loss_weight` | 0.1 | Contrastive loss weight |

### Modifying Hyperparameters

Edit the YAML configuration files in `src/configs/model/` to change parameters.

For example, to adjust the mixing ratio in RFFREEDOM:

```yaml
# src/configs/model/RFFREEDOM.yaml
rf_mix_ratio: 0.2                    # Changed from 0.1
rf_inference_mix_ratio: 0.3          # Changed from 0.2
```

## Troubleshooting

### Model Import Errors

If you see "No module named 'models.rffreedom'":
- Ensure you're in the `src/` directory when running
- Check that model files are lowercase: `rffreedom.py`, not `rf_freedom.py`

### CUDA Out of Memory

If you encounter OOM errors:
1. Reduce batch size in the model config
2. Reduce `rf_hidden_dim` from 128 to 64
3. Reduce `rf_n_layers` from 2 to 1

### RF Not Training

If RF loss shows as 0.0:
- Check that `use_rf: True` in config
- Verify `rf_warmup_epochs` is not too high
- Ensure multimodal features (image/text) are available

## Expected Results

Based on RFGUME experiments, you should expect:

- **NDCG@10**: +1% to +3% improvement
- **Recall@20**: +1% to +3% improvement
- **Precision@5**: +1% to +2% improvement

Results may vary by dataset and model.

## Log Files

Training logs are saved to:
```
log/RFFREEDOM-{dataset}.txt
log/RFBM3-{dataset}.txt
log/RFLGMRec-{dataset}.txt
```

Look for RF training messages:
```
[RF Train] epoch=5, rf_loss=0.123456, cl_loss=0.234567
```

## Advanced Usage

### Single Epoch Test

For quick testing:

```bash
cd src
python main.py --model RFFREEDOM --dataset baby --epochs 1
```

### Disable RF Enhancement

To run with RF disabled (equivalent to original model):

```yaml
# In config file
use_rf: False
```

Or create a custom config:

```bash
python main.py --model RFFREEDOM --dataset baby --config_dict "{'use_rf': False}"
```

### Custom Experiment

Create your own experiment script:

```python
from utils.utils import get_model
from utils.dataset import RecDataset

# Load model
config = {...}  # Your config
dataset = RecDataset(config)
model_class = get_model('RFFREEDOM')
model = model_class(config, dataset)

# Train as usual
```

## Citation

If you use these RF-enhanced models in your research, please cite:

```bibtex
@inproceedings{lgmrec2024,
  title={LGMRec: Local and Global Graph Learning for Multimodal Recommendation},
  author={...},
  booktitle={AAAI},
  year={2024}
}

@inproceedings{freedom2022,
  title={FREEDOM: A Tale of Two Graphs: Freezing and Denoising Graph Structures for Multimodal Recommendation},
  author={...},
  booktitle={...},
  year={2022}
}

@inproceedings{bm3,
  title={Bootstrap Latent Representations for Multi-modal Recommendation},
  author={...},
  year={2022}
}
```

## Support

For detailed implementation information, see:
- `RF_IMPLEMENTATION_SUMMARY.md` - Complete technical documentation
- Original plan: `~/.claude/plans/dynamic-strolling-lollipop.md`

For issues or questions:
- Check the troubleshooting section above
- Review the implementation summary document
- Verify your configuration files match the examples

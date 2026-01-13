# RF Module Integration Implementation Summary

## Overview

Successfully integrated Rectified Flow (RF) modules into three recommendation models: FREEDOM, BM3, and LGMRec. This implementation follows the design patterns established in RFGUME and creates independent RF-enhanced versions for performance comparison.

## Implementation Date

**Completed**: January 13, 2026

## Created Files

### 1. Model Implementations

#### RFFREEDOM (`src/models/rffreedom.py`)
- **Base Model**: FREEDOM
- **Target Embeddings**: `i_g_embeddings` - GCN-aggregated item embeddings from user-item interaction graph
- **Conditions**:
  - Image features: `image_trs(image_embedding.weight)`
  - Text features: `text_trs(text_embedding.weight)`
- **Enhancement Location**: After collaborative graph convolution, before multimodal feature fusion
- **Lines of Code**: ~120 lines

#### RFBM3 (`src/models/rfbm3.py`)
- **Base Model**: BM3
- **Target Embeddings**: `i_g_embeddings_ori` - GCN-aggregated item embeddings (before residual connection)
- **Conditions**:
  - Visual features: `image_trs(image_embedding.weight)`
  - Text features: `text_trs(text_embedding.weight)`
- **Enhancement Location**: After GCN output, before Predictor layer
- **Lines of Code**: ~115 lines

#### RFLGMRec (`src/models/rflgmrec.py`)
- **Base Model**: LGMRec
- **Target Embeddings**: `i_cge` - Item collaborative graph embeddings
- **Conditions**:
  - Visual features: `mge('v')` - MGE visual features (item part only)
  - Text features: `mge('t')` - MGE text features (item part only)
- **Enhancement Location**: In MGE layer, enhancing collaborative graph embeddings
- **Lines of Code**: ~140 lines

### 2. Configuration Files

#### RFFREEDOM.yaml (`src/configs/model/RFFREEDOM.yaml`)
- Inherits all FREEDOM hyperparameters
- RF parameters:
  - `rf_hidden_dim`: 128
  - `rf_n_layers`: 2
  - `rf_learning_rate`: 0.0001
  - `rf_sampling_steps`: 10
  - `rf_warmup_epochs`: 5
  - `rf_mix_ratio`: 0.1 (training)
  - `rf_inference_mix_ratio`: 0.2 (inference)
  - `rf_contrast_temp`: 0.2
  - `rf_loss_weight`: 0.1

#### RFBM3.yaml (`src/configs/model/RFBM3.yaml`)
- Inherits all BM3 hyperparameters
- Same RF parameters as RFFREEDOM

#### RFLGMRec.yaml (`src/configs/model/RFLGMRec.yaml`)
- Inherits all LGMRec hyperparameters
- Includes dataset-specific parameters (baby/sports/clothing)
- Same RF parameters as RFFREEDOM

### 3. Experiment Scripts

#### run_rf_comparison.sh (`run_rf_comparison.sh`)
- Bash script to run comprehensive comparison experiments
- Executes both original and RF-enhanced versions
- Tests on three datasets: Baby, Clothing, Sports
- Automated experiment workflow

#### compare_results.py (`compare_results.py`)
- Python script for results analysis
- Parses log files and extracts metrics
- Generates comparison tables with improvements
- Creates summary statistics
- Outputs CSV files for further analysis

## Design Principles

### Consistent with RFGUME Pattern

All implementations follow the same RF usage pattern:

1. **Target**: Collaborative filtering embeddings (ID-based, graph-aggregated)
2. **Conditions**: Multimodal features (image + text)
3. **Generation**: From Gaussian noise through ODE sampling
4. **Module**: Unified `RFEmbeddingGenerator` from `rf_modules.py`

### Key Features

1. **Independent RF Training**:
   - RF module has its own optimizer
   - Independent learning rate
   - Separate backpropagation

2. **Warmup Strategy**:
   - First 5 epochs: use original embeddings only
   - After warmup: gradually mix RF-generated embeddings

3. **Two-Phase Loss**:
   - Rectified Flow loss: velocity field matching
   - Contrastive loss: embedding alignment

4. **Training/Inference Modes**:
   - Training: 10% RF mixing ratio
   - Inference: 20% RF mixing ratio

## Verification

### Import Tests

All models successfully imported and loaded:

```
✓ RFFREEDOM loaded successfully
✓ RFBM3 loaded successfully
✓ RFLGMRec loaded successfully
```

### Dynamic Loading Test

All models successfully loaded through the system's dynamic model loading mechanism:

```python
from utils.utils import get_model
model_class = get_model('RFFREEDOM')  # ✓ Success
model_class = get_model('RFBM3')      # ✓ Success
model_class = get_model('RFLGMRec')   # ✓ Success
```

## How to Use

### Running Individual Models

```bash
cd GenMMRec/src

# Run RF-enhanced FREEDOM on Baby dataset
python main.py --model RFFREEDOM --dataset baby

# Run RF-enhanced BM3 on Clothing dataset
python main.py --model RFBM3 --dataset clothing

# Run RF-enhanced LGMRec on Sports dataset
python main.py --model RFLGMRec --dataset sports
```

### Running Comparison Experiments

```bash
cd GenMMRec

# Run all comparisons (original vs RF-enhanced)
./run_rf_comparison.sh

# Analyze results
python compare_results.py --datasets baby clothing sports --models FREEDOM BM3 LGMRec
```

### Output Files

- **Log files**: `GenMMRec/log/{MODEL}-{DATASET}.txt`
- **Results**: `rf_comparison_results.csv`
- **Summary**: `rf_comparison_results_summary.csv`

## Expected Performance

Based on RFGUME results, RF-enhanced versions are expected to show:

- **NDCG@10**: 1-3% improvement
- **Recall@20**: 1-3% improvement
- **Precision@5**: 1-2% improvement

Performance improvements should be consistent across Baby, Clothing, and Sports datasets.

## Technical Details

### RF Module Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `embedding_dim` | 64 | Base embedding dimension |
| `rf_hidden_dim` | 128 | RF network hidden dimension |
| `rf_n_layers` | 2 | Number of residual blocks |
| `rf_dropout` | 0.1 | Dropout rate |
| `rf_learning_rate` | 0.0001 | Independent learning rate |
| `rf_sampling_steps` | 10 | ODE sampling steps |
| `rf_warmup_epochs` | 5 | Warmup epochs |
| `rf_mix_ratio` | 0.1 | Training mix ratio |
| `rf_inference_mix_ratio` | 0.2 | Inference mix ratio |
| `rf_contrast_temp` | 0.2 | Contrastive loss temperature |
| `rf_loss_weight` | 0.1 | Contrastive loss weight |

### RF Enhancement Locations

**RFFREEDOM**:
- Line ~53 in forward(): After GCN, before fusion

**RFBM3**:
- Line ~70 in forward(): After GCN, before predictor

**RFLGMRec**:
- Line ~64 in forward(): After CGE, before MGE fusion

## Architecture Diagram

```
Original Model:
  ID Embeddings → GCN → [Target] → ... → Final Embeddings

RF-Enhanced Model:
  ID Embeddings → GCN → [Target] ──┬──→ Mix → ... → Final Embeddings
                                    │
  Multimodal Features → [Conditions] → RF Generator ──┘
                                         ↑
                                    ODE Sampling
```

## Dependencies

All RF-enhanced models depend on:

- `models.rf_modules.RFEmbeddingGenerator`
- Base model classes (`FREEDOM`, `BM3`, `LGMRec`)
- PyTorch >= 1.8.0

No additional dependencies required beyond the base GenMMRec environment.

## Notes

1. **File Naming**: Model files must be lowercase (`rffreedom.py`, not `rf_freedom.py`) for dynamic loading
2. **Set Epoch**: All RF models implement `set_epoch()` method for proper warmup control
3. **Logging**: RF loss is logged once per epoch to avoid cluttering output
4. **Detachment**: Target and condition embeddings are detached before RF training to prevent gradient leakage

## Future Work

Possible extensions:

1. **Hyperparameter Tuning**: Explore optimal RF mixing ratios per dataset
2. **Ablation Studies**: Test impact of different RF components
3. **Other Models**: Extend RF enhancement to MMGCN, LATTICE, etc.
4. **User Embeddings**: Also enhance user embeddings with RF

## Contact

For questions or issues, refer to the original plan document:
`~/.claude/plans/dynamic-strolling-lollipop.md`

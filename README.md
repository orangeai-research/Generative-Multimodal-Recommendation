# Generative-Multimodal-Recommendation (青云)
> “好风凭借力，送我上青云”——《临江仙・柳絮》
<div align="center">
  <a href=""><img width="300px" height="auto" src="./GenMMRec/images/qingyun.png"></a>
</div>

> Awesome Research on Generative Multimodal Recommendation: A Comprehensive Survey
> 
![GenRec](https://img.shields.io/badge/Survey-GenMMRec-orange) ![License](https://img.shields.io/badge/License-MIT-B39DDB) ![python](https://img.shields.io/badge/python-3.8+-blue) ![pytorch](https://img.shields.io/badge/pytorch-2.0-FFA000) ![Stars](https://img.shields.io/github/stars/orangeheyue/Generative-Multimodal-Recommendation?style=social)

# Weekly Dev Log
### News: 本周更新模型如下：
## 🚀 **[Update ] **
- **新增模型**: 2026-01-05新增DiffRec生成式模型, wandb可视化组件 
- **新增模型**: 2026-01-04新增CoDMR生成式模型。
- **新增模型**: 2026-01-03新增LD4MRec生成式模型。
- **新增模型**: 2025-12-31引入了新的Rectify Flow 机制，已打通RFMRec模型的初步流程。
- **新增模型**: 2025-12-30新增GenRec-V1生成式模型到当前框架中，模型、配置文件、数据读取、训练代码已验证。

| **Model**       | **Paper**                                                                                             | **Conference/Journal** | **Code**    |
|------------------|--------------------------------------------------------------------------------------------------------|------------------------|-------------|
| **Newly added**  |                                                                                                        |                        |             |
| CoDMR         | [ Collaborative Diffusion Models for Recommendation](https://dl.acm.org/doi/pdf/10.1145/3726302.3729929)                                          | SIGIR'25                 | codmr.py |
| GenRec-V1 | [Flip is Better than Noise: Unbiased Interest Generation for Multimedia Recommendation](https://dl.acm.org/doi/abs/10.1145/3746027.3755743)                                 | MM'25                  | genrecv1.py  |
| DiffMM     | [Diffmm: Multi-modal diffusion model for recommendation](https://arxiv.org/pdf/2406.11781)                       | MM'24               | diffmm.py          |
| LD4MRec  | [LD4MRec: Simplifying and Powering Diffusion Model for Multimedia Recommendation](https://arxiv.org/pdf/2309.15363)                                 | WWW'24                | ld4mrec.py  |
| DiffRec  | [DiffRec: Diffusion Recommender Model](https://arxiv.org/abs/2304.04971)                                 | SIGIR'23                | diffrec.py  |


### 📝 TODO / Next Week
- [1] 优化RFMRec模型。
- [2] 在数据集上测试本框架下的DiffMM, GenRec-V1的稳定性。


## Run the Code
1. Clone the repository
```bash
git clone 
```
2. Install dependencies
```bash
pip install -r requirements.txt
```
3. Run the code
```bash
cd GenMMRec/src
python main.py --model GenRecV1 --dataset baby
python main.py --model DiffMM --dataset baby
or
python run.py --config configs/diffmm.yaml
```



## Survey Papers
- [A Review of Modern Recommender Systems Using Generative Models (Gen-RecSys)](https://dl.acm.org/doi/pdf/10.1145/3637528.3671474), KDD 24

- [Multi-modal Generative Models in Recommendation System](https://arxiv.org/pdf/2409.10993), 2024

- [Multimodal Pretraining, Adaptation, and Generation for Recommendation: A Survey](https://arxiv.org/pdf/2404.00621), KDD 24




## Research Papers
- [Generative Recommendation: Towards Personalized Multimodal Content Generation](https://dl.acm.org/doi/pdf/10.1145/3701716.3717529), WWW 25
- [MULTIMODAL QUANTITATIVE LANGUAGE FOR GENERATIVE RECOMMENDATION](https://arxiv.org/pdf/2504.05314?), ICLR 2025
- [TOWARDS UNIFIED MULTI-MODAL PERSONALIZATION: LARGE VISION-LANGUAGE MODELS FOR GENERATIVE RECOMMENDATION AND BEYOND](https://arxiv.org/pdf/2403.10667?), ICLR 2024

## Generative Multimodal Top-K Recommendation
- [Multimodal Conditioned Diffusion Model for Recommendation](https://web.archive.org/web/20240521123350id_/https://dl.acm.org/doi/pdf/10.1145/3589335.3651956), WWW 2024  MCDRec

## Generative Multimodal Sequential Recommendation
- [Harnessing Multimodal Large Language Models for Multimodal Sequential Recommendation](https://ojs.aaai.org/index.php/AAAI/article/view/33426), AAAI 2025

## Generative Multimodal CTR Recommendation


## Generative Multimodal POI Recommendation 


## Generative Multimodal Food Recommendation


## Generative Multimodal Medicine Recommendation


## Education Videoes

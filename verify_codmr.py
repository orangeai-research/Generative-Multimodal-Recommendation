import sys
import os
import torch
from unittest.mock import MagicMock

# Add src to path
sys.path.append(os.path.abspath('GenMMRec/src'))

# Mock Config
config = {
    'embedding_size': 32,
    'uiLayers': 2,
    'au_uiLayers': 2,
    'learning_rate': 0.01,
    'reg_weight': 1e-4,
    'ssl_temp': 0.5,
    'steps': 5,
    'sampling_steps': 5,
    'mean_type': 'eps',
    'mean_typeNon': 'x0',
    'noise_schedule': 'linear-var',
    'noise_scale': 0.1,
    'noise_min': 0.0001,
    'noise_max': 0.02,
    'reweight': True,
    'mlp_dims': [8],
    'latent_size': 8,
    'emb_size': 10,
    'norm': False,
    'mlp_act_func': 'tanh',
    'knn_k': 10,
    'elbo_w': 0.1,
    'di_pre_w': 0.1,
    'con_fe_w': 0.1,
    'ssl_reg': 0.1,
    'device': 'cpu',
    'USER_ID_FIELD': 'user_id',
    'ITEM_ID_FIELD': 'item_id',
    'NEG_PREFIX': 'neg_',
    'train_batch_size': 256,
    'end2end': False,
    'is_multimodal_model': False,
    'data_path': './',
    'dataset': 'mock',
    'vision_feature_file': 'v_feat.npy',
    'text_feature_file': 't_feat.npy'
}

# Mock Dataset
class MockDataset:
    def __init__(self):
        self.n_users = 10
        self.n_items = 5
        self.dataset = self # Self-reference to mock DataLoader.dataset structure
    
    def get_user_num(self):
        return self.n_users
        
    def get_item_num(self):
        return self.n_items
        
    def inter_matrix(self, form='coo'):
        import scipy.sparse as sp
        return sp.coo_matrix((np.ones(5), (np.arange(5), np.arange(5))), shape=(10, 5))

import numpy as np
dataset = MockDataset()

# Import Model
from models.codmr import CoDMR

print("Initializing CoDMR...")
model = CoDMR(config, dataset)
print("CoDMR Initialized.")

# Test Forward
print("Testing Calculate Loss...")
interaction = [torch.tensor([0, 1]), torch.tensor([0, 1]), torch.tensor([1, 0])]
loss = model.calculate_loss(interaction)
print(f"Loss: {loss.item()}")

# Test Predict
print("Testing Predict...")
scores = model.full_sort_predict(interaction)
print(f"Scores shape: {scores.shape}")

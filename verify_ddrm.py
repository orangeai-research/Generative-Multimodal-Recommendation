import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add GenMMRec/src to path
sys.path.append(os.path.abspath('GenMMRec/src'))

from models.ddrm import DDRM

# Mock Dataset
class MockDatasetInner:
    def __init__(self):
        self.n_users = 10
        self.n_items = 5
        
    def get_user_num(self):
        return self.n_users
        
    def get_item_num(self):
        return self.n_items
        
    def inter_matrix(self, form='coo'):
        import scipy.sparse as sp
        row = np.array([0, 1, 2, 0])
        col = np.array([0, 1, 2, 3])
        data = np.ones(4)
        return sp.coo_matrix((data, (row, col)), shape=(10, 5))

class MockDataLoader:
    def __init__(self):
        self.dataset = MockDatasetInner()

# Mock Config
config = {
    'embedding_size': 8,
    'lightGCN_n_layers': 2,
    'keep_prob': 0.8,
    'A_split': False,
    'dropout': False,
    'steps': 5,
    'noise_scale': 0.001,
    'noise_min': 0.0001,
    'noise_max': 0.01,
    'reg_weight': 0.01,
    'dims': [16],
    'norm': True,
    'act': 'tanh',
    'alpha': 0.1,
    'beta': 0.1,
    'noise_schedule': 'linear-var',
    'device': 'cpu',
    'sampling_steps': 5, # Must match steps or be smaller, but usually same as steps for full inference
    'sampling_noise': False,
    'USER_ID_FIELD': 'user_id',
    'ITEM_ID_FIELD': 'item_id',
    'NEG_PREFIX': 'neg_',
    'neg_sampling': None,
    'train_batch_size': 1024,
    'end2end': True,
    'is_multimodal_model': False,
    'reweight': True
}

def verify():
    print("Initializing DDRM...")
    dataloader = MockDataLoader()
    model = DDRM(config, dataloader)
    print("DDRM Initialized.")
    
    print("Testing Calculate Loss...")
    # interaction: [users, pos_items, neg_items]
    interaction = [torch.tensor([0, 1]), torch.tensor([0, 1]), torch.tensor([1, 0])]
    loss = model.calculate_loss(interaction)
    print(f"Loss: {loss.item()}")
    
    print("Testing Predict...")
    scores = model.full_sort_predict(interaction)
    print(f"Scores shape: {scores.shape}") # Should be [2, 5]
    
    print("Verification Passed.")

if __name__ == '__main__':
    verify()

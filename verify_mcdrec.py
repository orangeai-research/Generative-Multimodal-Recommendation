import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add GenMMRec/src to path
sys.path.append(os.path.abspath('GenMMRec/src'))
sys.path.append(os.path.abspath('.'))

from GenMMRec.src.models.mcdrec import MCDRec

# Mock Dataset
class MockDatasetInner:
    def __init__(self):
        self.n_users = 10
        self.n_items = 64 # Must be square e.g. 64=8x8
        
    def get_user_num(self):
        return self.n_users
        
    def get_item_num(self):
        return self.n_items
        
    def inter_matrix(self, form='coo'):
        import scipy.sparse as sp
        row = np.array([0, 1, 2, 0])
        col = np.array([0, 1, 2, 3])
        data = np.ones(4)
        return sp.coo_matrix((data, (row, col)), shape=(self.n_users, self.n_items))

class MockDataLoader:
    def __init__(self):
        self.dataset = MockDatasetInner()

# Mock Config
config = {
    'embedding_size': 64, # 8x8
    'lightGCN_n_layers': 2,
    'dropout': 0.0,
    'keep_prob': 0.8,
    'A_split': False,
    'steps': 5,
    'noise_scale': 0.001,
    'noise_min': 0.0001,
    'noise_max': 0.01,
    'lambda_dm': 0.1,
    'tau': 0.5,
    'rho': 0.1,
    'learning_rate': 0.001,
    'weight_decay': 0.0001,
    'device': 'cpu',
    'USER_ID_FIELD': 'user_id',
    'ITEM_ID_FIELD': 'item_id',
    'NEG_PREFIX': 'neg_',
    'neg_sampling': None,
    'train_batch_size': 1024,
    'end2end': True,
    'is_multimodal_model': False
}

def verify():
    print("Initializing MCDRec...")
    dataloader = MockDataLoader()
    model = MCDRec(config, dataloader)
    
    # Inject fake multimodal features
    model.v_feat = torch.randn(64, 32)
    model.t_feat = torch.randn(64, 32)
    model.v_mlp = nn.Linear(32, 64)
    model.t_mlp = nn.Linear(32, 64)
    
    print("MCDRec Initialized.")
    
    print("Testing Pre-Epoch Processing (Graph Denoising)...")
    model.pre_epoch_processing()
    print("Graph Denoising Done.")
    
    print("Testing Calculate Loss...")
    # interaction: [users, pos_items, neg_items]
    interaction = [torch.tensor([0, 1]), torch.tensor([0, 1]), torch.tensor([1, 0])]
    loss = model.calculate_loss(interaction)
    print(f"Loss: {loss.item()}")
    
    print("Testing Predict...")
    scores = model.full_sort_predict(interaction)
    print(f"Scores shape: {scores.shape}") # Should be [2, 64]
    
    print("Verification Passed.")

if __name__ == '__main__':
    verify()

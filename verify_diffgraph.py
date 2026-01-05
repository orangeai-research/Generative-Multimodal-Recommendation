import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add GenMMRec/src to path
sys.path.append(os.path.abspath('GenMMRec/src'))

from models.diffgraph import DiffGraph

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
    'gcn_layer': 2,
    'steps': 5,
    'noise_scale': 0.001,
    'noise_min': 0.0001,
    'noise_max': 0.01,
    'reg_weight': 0.01,
    'dims': [16],
    'd_emb_size': 4,
    'norm': True,
    'dropout': 0.1,
    'device': 'cpu',
    'sampling_steps': 0,
    'USER_ID_FIELD': 'user_id',
    'ITEM_ID_FIELD': 'item_id',
    'NEG_PREFIX': 'neg_',
    'neg_sampling': None,
    'train_batch_size': 1024,
    'end2end': True, # Skip feature loading
    'is_multimodal_model': False
}

def verify():
    print("Initializing DiffGraph...")
    dataloader = MockDataLoader()
    # DiffGraph expects dataset (not dataloader) in __init__ if we follow typical RecBole pattern, 
    # BUT GenMMRec's GeneralRecommender expects dataloader in init.
    # However, DiffGraph.__init__ calls super(..., dataset).
    # In GenMMRec/src/models/diffgraph.py: super(DiffGraph, self).__init__(config, dataset)
    # So 'dataset' passed to DiffGraph must be the dataloader expected by GeneralRecommender.
    model = DiffGraph(config, dataloader)
    print("DiffGraph Initialized.")
    
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

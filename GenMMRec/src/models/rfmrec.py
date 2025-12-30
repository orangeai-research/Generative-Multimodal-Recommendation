# coding: utf-8
# Rectified Flow for Multimodal Recommendation (RFMRec)
"""
Reference:
    Liu et al. "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow"
    arXiv 2022

Implementation of Rectified Flow for generating user/item embeddings in multimodal recommendation.
Key features:
    - Fast sampling with straight paths (1-step after reflow)
    - Multi-scale conditional injection
    - Collaborative signal guidance (user-item interaction, multimodal features)
"""

import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender


class RFMREC(GeneralRecommender):
    """
    Rectified Flow for Multimodal Recommendation

    This model generates user/item embeddings using Rectified Flow ODE,
    conditioned on collaborative signals (interactions, multimodal features).
    """

    def __init__(self, config, dataset):
        super(RFMREC, self).__init__(config, dataset)

        # Model hyperparameters
        self.embedding_dim = config['embedding_size']
        self.hidden_dim = config['rf_hidden_dim']
        self.n_layers = config['rf_n_layers']
        self.dropout = config['rf_dropout']
        self.reg_weight = config['reg_weight']

        # Rectified Flow specific parameters
        self.n_sampling_steps = config['rf_sampling_steps']  # ODE solver steps
        self.use_reflow = config['rf_use_reflow']
        self.reflow_steps = config['rf_reflow_steps']  # Number of reflow iterations

        # Loss weights
        self.lambda_rf = config['lambda_rf']  # Rectified Flow loss weight
        self.lambda_rec = config['lambda_rec']  # Reconstruction loss weight

        # Load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)

        # Base embeddings (trainable)
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        # Multimodal feature processing
        self.has_visual = self.v_feat is not None
        self.has_text = self.t_feat is not None

        if self.has_visual:
            self.visual_proj = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
        if self.has_text:
            self.text_proj = nn.Linear(self.t_feat.shape[1], self.embedding_dim)

        # Interaction graph processing
        self.norm_adj = self.get_norm_adj_mat()
        self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float().to(self.device)

        # Multi-scale Velocity Network (core of Rectified Flow)
        self.velocity_net = MultiScaleVelocityNet(
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            dropout=self.dropout,
            # Condition dimensions
            interaction_dim=self.embedding_dim * 2,  # user + item
            visual_dim=self.embedding_dim if self.has_visual else 0,
            text_dim=self.embedding_dim if self.has_text else 0,
        )

        print(f"RFMRec initialized: embedding_dim={self.embedding_dim}, "
              f"hidden_dim={self.hidden_dim}, n_layers={self.n_layers}")

    def get_norm_adj_mat(self):
        """Build normalized adjacency matrix for user-item interaction graph"""
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items),
                                dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.interaction_matrix.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()

        # Symmetric normalization: D^{-1/2} A D^{-1/2}
        rowsum = np.array(adj_mat.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        norm_adj = d_mat_inv_sqrt.dot(adj_mat).dot(d_mat_inv_sqrt)
        return norm_adj.tocsr()

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert scipy sparse matrix to torch sparse tensor"""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def graph_convolution(self, embeddings, n_layers=2):
        """Simple graph convolution for extracting collaborative signals"""
        all_embeddings = [embeddings]
        for _ in range(n_layers):
            embeddings = torch.sparse.mm(self.norm_adj, embeddings)
            all_embeddings.append(embeddings)

        # Mean pooling over all layers
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1)

        return all_embeddings

    def get_collaborative_conditions(self, users, items):
        """
        Extract multi-scale collaborative signals as conditions

        Returns:
            dict of condition tensors
        """
        conditions = {}
        batch_size = users.shape[0]

        # 1. User-item interaction signal (from graph convolution)
        all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        graph_embeddings = self.graph_convolution(all_embeddings, n_layers=2)

        user_graph_emb = graph_embeddings[:self.n_users]
        item_graph_emb = graph_embeddings[self.n_users:]

        # Concatenate user and item embeddings as interaction condition
        conditions['interaction'] = torch.cat([
            user_graph_emb[users],
            item_graph_emb[items]
        ], dim=-1)

        # 2. Visual features
        if self.has_visual:
            visual_feat = self.visual_proj(self.v_feat[items])
            conditions['visual'] = visual_feat

        # 3. Text features
        if self.has_text:
            text_feat = self.text_proj(self.t_feat[items])
            conditions['text'] = text_feat

        return conditions

    def rectified_flow_loss(self, users, items):
        """
        Compute Rectified Flow loss (Equation 1 in paper)

        Loss = E_t [ ||v(X_t, t, c) - (X1 - X0)||^2 ]
        where X_t = t*X1 + (1-t)*X0
        """
        batch_size = users.shape[0]

        # Get collaborative conditions
        conditions = self.get_collaborative_conditions(users, items)

        # Target embeddings X1 (from current model state)
        user_emb = self.user_embedding(users)
        item_emb = self.item_embedding(items)
        X1 = torch.cat([user_emb, item_emb], dim=-1)  # Shape: (batch, 2*embedding_dim)

        # Random noise X0
        X0 = torch.randn_like(X1)

        # Random time step t ~ Uniform[0, 1]
        t = torch.rand(batch_size, 1).to(self.device)

        # Linear interpolation: X_t = t*X1 + (1-t)*X0
        X_t = t * X1 + (1 - t) * X0

        # Predict velocity: v(X_t, t, conditions)
        v_pred = self.velocity_net(X_t, t, conditions)

        # Target velocity: X1 - X0 (direction of straight path)
        v_target = X1 - X0

        # Rectified Flow loss (MSE)
        rf_loss = F.mse_loss(v_pred, v_target)

        return rf_loss

    def forward(self, users, items):
        """
        Forward pass for generating embeddings

        Args:
            users: user indices, shape (batch,)
            items: item indices, shape (batch,)

        Returns:
            user_embeddings, item_embeddings
        """
        # Get collaborative conditions
        conditions = self.get_collaborative_conditions(users, items)

        # Sample from noise
        batch_size = users.shape[0]
        z_0 = torch.randn(batch_size, self.embedding_dim * 2).to(self.device)

        # Solve ODE to generate embeddings
        # After reflow, we can use just 1 step!
        n_steps = 1 if self.use_reflow else self.n_sampling_steps
        z_1 = self.sample_ode(z_0, conditions, n_steps=n_steps)

        # Split into user and item embeddings
        user_emb, item_emb = z_1.chunk(2, dim=-1)

        return user_emb, item_emb

    def sample_ode(self, z_0, conditions, n_steps=100):
        """
        Solve ODE using Euler method

        dz_t = v(z_t, t, conditions) dt
        """
        z_t = z_0
        dt = 1.0 / n_steps

        for i in range(n_steps):
            t = torch.full((z_0.shape[0], 1), i * dt).to(self.device)
            v = self.velocity_net(z_t, t, conditions)
            z_t = z_t + v * dt

        return z_t

    def calculate_loss(self, interaction):
        """
        Calculate training loss

        Total loss = Rectified Flow loss + BPR loss + regularization
        """
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        # 1. Rectified Flow loss (main loss)
        rf_loss = self.rectified_flow_loss(users, pos_items)

        # 2. BPR loss for recommendation task
        user_emb = self.user_embedding(users)
        pos_emb = self.item_embedding(pos_items)
        neg_emb = self.item_embedding(neg_items)

        pos_scores = torch.sum(user_emb * pos_emb, dim=1)
        neg_scores = torch.sum(user_emb * neg_emb, dim=1)

        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))

        # 3. Regularization
        reg_loss = self.reg_weight * (
            user_emb.norm(2).pow(2) + pos_emb.norm(2).pow(2) + neg_emb.norm(2).pow(2)
        ) / users.shape[0]

        # Total loss
        total_loss = self.lambda_rf * rf_loss + self.lambda_rec * bpr_loss + reg_loss

        return total_loss

    def full_sort_predict(self, interaction):
        """
        Predict scores for all items (for evaluation)
        """
        user = interaction[0]

        # Use base embeddings for efficient full-sort prediction
        user_emb = self.user_embedding(user)
        all_item_emb = self.item_embedding.weight

        scores = torch.matmul(user_emb, all_item_emb.transpose(0, 1))

        return scores


class MultiScaleVelocityNet(nn.Module):
    """
    Multi-scale Velocity Network with conditional injection

    Architecture:
        - Time embedding (sinusoidal encoding)
        - Condition encoders for each modality
        - ResNet-like blocks with FiLM modulation
    """

    def __init__(self, embedding_dim, hidden_dim, n_layers, dropout,
                 interaction_dim, visual_dim, text_dim):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Time embedding (positional encoding)
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(128),
            nn.Linear(128, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Condition encoders
        self.condition_encoders = nn.ModuleDict()

        if interaction_dim > 0:
            self.condition_encoders['interaction'] = nn.Sequential(
                nn.Linear(interaction_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )

        if visual_dim > 0:
            self.condition_encoders['visual'] = nn.Sequential(
                nn.Linear(visual_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )

        if text_dim > 0:
            self.condition_encoders['text'] = nn.Sequential(
                nn.Linear(text_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )

        # Input projection
        self.input_proj = nn.Linear(embedding_dim * 2 + hidden_dim, hidden_dim)

        # Residual blocks with conditional modulation
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim, condition_dim=hidden_dim, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, embedding_dim * 2)

    def forward(self, x, t, conditions):
        """
        Args:
            x: current state X_t, shape (batch, embedding_dim*2)
            t: time step, shape (batch, 1)
            conditions: dict of condition tensors

        Returns:
            velocity v(x, t, c), shape (batch, embedding_dim*2)
        """
        # Time embedding
        t_emb = self.time_embed(t)

        # Aggregate all conditions
        cond_emb = t_emb
        for name, encoder in self.condition_encoders.items():
            if name in conditions:
                cond_emb = cond_emb + encoder(conditions[name])

        # Input: [x, t_emb]
        h = torch.cat([x, t_emb], dim=-1)
        h = self.input_proj(h)

        # Residual blocks with condition injection
        for block in self.res_blocks:
            h = block(h, cond_emb)

        # Output velocity
        v = self.output_proj(h)

        return v


class ResidualBlock(nn.Module):
    """
    Residual block with FiLM (Feature-wise Linear Modulation) conditioning
    """

    def __init__(self, in_dim, out_dim, condition_dim, dropout=0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )

        # FiLM: scale and shift modulation
        self.cond_scale = nn.Linear(condition_dim, out_dim)
        self.cond_shift = nn.Linear(condition_dim, out_dim)

        # Skip connection
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x, cond):
        """
        Args:
            x: input features, shape (batch, in_dim)
            cond: condition features, shape (batch, condition_dim)

        Returns:
            output features, shape (batch, out_dim)
        """
        h = self.net(x)

        # FiLM modulation: h = scale * h + shift
        scale = self.cond_scale(cond)
        shift = self.cond_shift(cond)
        h = scale * h + shift

        # Residual connection
        return h + self.skip(x)


class SinusoidalPositionEmbedding(nn.Module):
    """
    Sinusoidal positional encoding for time steps
    (Similar to Transformer positional encoding)
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        Args:
            t: time steps, shape (batch, 1)

        Returns:
            positional embeddings, shape (batch, dim)
        """
        device = t.device
        half_dim = self.dim // 2

        embeddings = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t * embeddings[None, :]

        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)

        return embeddings

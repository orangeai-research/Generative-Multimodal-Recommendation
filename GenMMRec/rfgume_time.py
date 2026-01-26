# Rectified Flow for GUME: RF-GUME
# Integrating Rectified Flow into GUME for generating extended_id_embeds

import os
import numpy as np
import scipy.sparse as sp
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.gume import GUME


class SimpleVelocityNet(nn.Module):
    """
    简单的速度场网络，用于Rectified Flow

    Args:
        embedding_dim: 嵌入维度
        hidden_dim: 隐藏层维度
        n_layers: 网络层数
        dropout: dropout率
        condition_dim: 条件维度（image + text）
    """

    def __init__(self, embedding_dim, hidden_dim, n_layers, dropout, condition_dim):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # 时间嵌入层（使用正弦位置编码）
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(64),
            nn.Linear(64, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # 条件编码器
        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # 残差块
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, hidden_dim, dropout) for _ in range(n_layers)]
        )

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x, t, conditions):
        """
        Args:
            x: 当前状态 X_t, shape (batch, embedding_dim)
            t: 时间步, shape (batch, 1)
            conditions: 条件特征（image + text）, shape (batch, condition_dim)

        Returns:
            velocity: 速度 v(x, t, c), shape (batch, embedding_dim)
        """
        # 时间嵌入
        t_emb = self.time_embed(t)  # (batch, hidden_dim)

        # 条件嵌入
        cond_emb = self.condition_encoder(conditions)  # (batch, hidden_dim)

        # 输入投影
        h = self.input_proj(x)  # (batch, hidden_dim)

        # 融合时间和条件
        h = h + t_emb + cond_emb

        # 通过残差块
        for res_block in self.res_blocks:
            h = res_block(h)

        # 输出速度
        v = self.output_proj(h)

        return v


class ResidualBlock(nn.Module):
    """简单的残差块"""

    def __init__(self, in_dim, out_dim, dropout):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        return F.silu(self.net(x) + self.skip(x))


class SinusoidalPositionEmbedding(nn.Module):
    """正弦位置编码（用于时间步）"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        Args:
            t: 时间步, shape (batch, 1)
        Returns:
            embeddings: shape (batch, dim)
        """
        device = t.device
        half_dim = self.dim // 2

        embeddings = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t * embeddings[None, :]

        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)

        return embeddings


class RFExtendedIdGenerator(nn.Module):
    """
    使用Rectified Flow生成extended_id_embeds

    这个模块独立于GUME的其他部分，便于进行消融实验
    """

    def __init__(self, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()

        self.embedding_dim = embedding_dim

        # 速度场网络（条件维度 = embedding_dim * 2，因为有image和text两个条件）
        self.velocity_net = SimpleVelocityNet(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            condition_dim=embedding_dim * 2,  # image + text conditions
        )

    def rectified_flow_loss(self, h1_embeds, h2_embeds, image_cond, text_cond):
        """
        Modified Rectified Flow loss with temporal graph augmentation

        Args:
            h1_embeds: Embeddings from historical graph (X0)
            h2_embeds: Embeddings from augmented graph (X1)
            image_cond: explicit_image_item condition
            text_cond: explicit_text_item condition

        Returns:
            rf_loss: Rectified Flow velocity loss
        """
        batch_size = h2_embeds.shape[0]

        # X0 = historical graph embeddings (NEW: was random noise)
        X0 = h1_embeds

        # X1 = augmented graph embeddings (NEW: was original graph)
        X1 = h2_embeds

        # Random time step t ~ Uniform[0, 1]
        t = torch.rand(batch_size, 1).to(X1.device)

        # Linear interpolation: X_t = t*X1 + (1-t)*X0
        X_t = t * X1 + (1 - t) * X0

        # Conditions
        conditions = torch.cat([image_cond, text_cond], dim=-1)

        # Predict velocity
        v_pred = self.velocity_net(X_t, t, conditions)

        # Target velocity: X1 - X0
        v_target = X1 - X0

        # RF loss
        rf_loss = F.mse_loss(v_pred, v_target)

        return rf_loss

    def sample_ode(self, h1_embeds, image_cond, text_cond, n_steps=10):
        """
        ODE sampling starting from historical embeddings (not random noise)

        Args:
            h1_embeds: Initial embeddings from historical graph
            image_cond, text_cond: Conditions
            n_steps: Number of Euler steps

        Returns:
            Generated augmented embeddings
        """
        z_t = h1_embeds  # Start from h1 (NEW: was random noise)
        dt = 1.0 / n_steps

        conditions = torch.cat([image_cond, text_cond], dim=-1)

        for i in range(n_steps):
            t = torch.full((h1_embeds.shape[0], 1), i * dt).to(h1_embeds.device)
            v = self.velocity_net(z_t, t, conditions)
            z_t = z_t + v * dt

        return z_t

    def forward(self, h1_embeds, image_cond, text_cond, n_steps=10):
        """
        Forward pass using historical embeddings as starting point

        Args:
            h1_embeds: Historical graph embeddings
            image_cond, text_cond: Conditions
            n_steps: ODE steps
        """
        generated_embeds = self.sample_ode(h1_embeds, image_cond, text_cond, n_steps)
        return generated_embeds


class RFGUME(GUME):
    """
    GUME with Rectified Flow

    在GUME的基础上，使用Rectified Flow生成extended_id_embeds
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # Store dataset reference for temporal splitting
        # Access the underlying RecDataset from DataLoader if needed
        self.dataset = dataset.dataset if hasattr(dataset, 'dataset') else dataset

        # RF相关参数
        self.use_rf = config["use_rf"] if "use_rf" in config else True
        self.rf_hidden_dim = (
            config["rf_hidden_dim"] if "rf_hidden_dim" in config else 256
        )
        self.rf_n_layers = config["rf_n_layers"] if "rf_n_layers" in config else 2
        self.rf_dropout = config["rf_dropout"] if "rf_dropout" in config else 0.1
        self.rf_sampling_steps = (
            config["rf_sampling_steps"] if "rf_sampling_steps" in config else 10
        )

        # 对比损失温度参数
        self.rf_contrast_temp = (
            config["rf_contrast_temp"] if "rf_contrast_temp" in config else 0.2
        )

        # RF Warmup: 前 N 个 epoch 不启用 RF，先让 GUME 收敛
        self.rf_warmup_epochs = (
            config["rf_warmup_epochs"] if "rf_warmup_epochs" in config else 0
        )
        self.current_epoch = 0  # 由 trainer 更新

        # RF 混合比例：0.0 = 纯GUME，1.0 = 纯RF
        self.rf_mix_ratio = config["rf_mix_ratio"] if "rf_mix_ratio" in config else 0.5

        # RF 推理时的混合比例
        self.rf_inference_mix_ratio = (
            config["rf_inference_mix_ratio"]
            if "rf_inference_mix_ratio" in config
            else 0.5
        )

        # RF学习率
        self.rf_learning_rate = (
            config["rf_learning_rate"] if "rf_learning_rate" in config else 0.001
        )

        # Reflow configuration (2-Rectified Flow)
        self.rf_reflow_mode = config['rf_reflow_mode'] if config['rf_reflow_mode'] in ['1-step', '2-step'] else '1-step'
        self.rf_reflow_epoch = config['rf_reflow_epoch'] if 'rf_reflow_epoch' in config else 50  # Epoch to switch to 2-RF
        self.rf_loss_weight = config['rf_loss_weight'] if 'rf_loss_weight' in config else 0.1  # Weight for contrastive loss

        # Temporal augmentation configuration
        self.temporal_ratio = config['temporal_ratio'] if 'temporal_ratio' in config else 0.0
        self.use_temporal_augmentation = self.temporal_ratio > 0

        # Initialize temporal graphs as None (built if temporal_ratio > 0)
        self.historical_adj = None
        self.augmented_adj = None
        self.time_ranges = None

        # Build temporal graphs if enabled
        if self.use_temporal_augmentation:
            self.build_temporal_graphs()

        if self.use_rf:
            # 使用单个统一的RF生成器处理用户和物品
            self.rf_generator = RFExtendedIdGenerator(
                embedding_dim=self.embedding_dim,
                hidden_dim=self.rf_hidden_dim,
                n_layers=self.rf_n_layers,
                dropout=self.rf_dropout,
            )

            # RF独立优化器
            self.rf_optimizer = torch.optim.AdamW(
                self.rf_generator.parameters(), lr=self.rf_learning_rate
            )

            # 用于控制每个epoch只打印一次日志
            self._rf_logged_this_epoch = False

            print(
                f"RF-GUME initialized: "
                f"hidden_dim={self.rf_hidden_dim}, n_layers={self.rf_n_layers}, "
                f"sampling_steps={self.rf_sampling_steps}, "
                f"warmup_epochs={self.rf_warmup_epochs}, "
                f"rf_lr={self.rf_learning_rate}, "
                f"mix_ratio(train/infer)={self.rf_mix_ratio}/{self.rf_inference_mix_ratio}"
            )

    def set_epoch(self, epoch):
        """由 trainer 调用，更新当前 epoch"""
        self.current_epoch = epoch
        if self.use_rf:
            self._rf_logged_this_epoch = False  # 重置日志标记

    def build_temporal_graphs(self):
        """Build historical and augmented adjacency matrices"""
        if not self.use_temporal_augmentation:
            return

        print(f"Building temporal graphs with temporal_ratio={self.temporal_ratio}")

        # Get temporal split
        historical_df, future_df, time_ranges = \
            self.dataset.get_temporal_split_interactions(self.temporal_ratio)

        print(f"Historical interactions: {len(historical_df)}, Future interactions: {len(future_df)}")

        # Build interaction matrix for historical data
        print("Building historical interaction matrix...")
        historical_inter = self._df_to_interaction_matrix(historical_df)

        print("Building historical adjacency matrix...")
        # Build historical adjacency using GUME's method (reuse existing logic)
        historical_adj_scipy = self._build_adj_mat_from_inter(historical_inter.tolil(), self.ii_adj.tolil())
        self.historical_adj = self.sparse_mx_to_torch_sparse_tensor(historical_adj_scipy).float().to(self.device)

        print("Reusing original adjacency as augmented graph...")
        # For augmented graph, reuse the original norm_adj (saves memory)
        # This is valid because norm_adj contains all interactions
        self.augmented_adj = self.norm_adj

        # Store time ranges
        self.time_ranges = time_ranges

        print("Temporal graphs built successfully")

    def _build_adj_mat_from_inter(self, interaction_lil, item_adj):
        """
        Build adjacency matrix from interaction matrix (based on GUME's get_adj_mat)

        Args:
            interaction_lil: Interaction matrix in lil format
            item_adj: Item-item adjacency in lil format

        Returns:
            Normalized adjacency matrix in csr format
        """
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()

        # Add user-item interactions
        adj_mat[:self.n_users, self.n_users:] = interaction_lil
        adj_mat[self.n_users:, :self.n_users] = interaction_lil.T

        # Add item-item similarity
        adj_mat[self.n_users:, self.n_users:] = item_adj

        adj_mat = adj_mat.todok()

        # Normalized adjacency (from GUME's get_adj_mat)
        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj = d_mat_inv.dot(adj)
            norm_adj = norm_adj.dot(d_mat_inv)
            return norm_adj.tocoo()

        norm_adj_mat = normalized_adj_single(adj_mat)
        return norm_adj_mat.tolil().tocsr()

    def _df_to_interaction_matrix(self, df):
        """Convert DataFrame to sparse interaction matrix"""
        if df.empty:
            return sp.coo_matrix((self.n_users, self.n_items), dtype=np.float32)

        users = df[self.dataset.uid_field].values
        items = df[self.dataset.iid_field].values
        data = np.ones(len(df))
        return sp.coo_matrix(
            (data, (users, items)),
            shape=(self.n_users, self.n_items),
            dtype=np.float32
        )

    def get_rf_sampling_steps(self):
        """Get number of ODE steps based on reflow mode and epoch"""
        if self.rf_reflow_mode == '1-step':
            return self.rf_sampling_steps
        elif self.rf_reflow_mode == '2-step':
            if self.current_epoch < self.rf_reflow_epoch:
                return self.rf_sampling_steps  # Stage 1: multi-step
            else:
                return 1  # Stage 2: single-step after reflow
        return self.rf_sampling_steps

    def is_rf_active(self):
        """检查 RF 是否已激活（RF始终训练，只是warmup期间不参与混合）"""
        return self.use_rf

    def forward(self, adj, train=False):
        """
        使用RF生成extended_id_embeds的前向传播

        Args:
            adj: 邻接矩阵
            train: 是否训练模式

        Returns:
            all_embeds: 最终的嵌入
            rf_outputs: RF相关的输出（用于计算损失）
            other_outputs: 其他输出
        """
        # ===== 原始GUME的多模态编码 =====
        image_item_embeds = torch.multiply(
            self.item_id_embedding.weight,
            self.image_space_trans(self.image_embedding.weight),
        )
        text_item_embeds = torch.multiply(
            self.item_id_embedding.weight,
            self.text_space_trans(self.text_embedding.weight),
        )

        item_embeds = self.item_id_embedding.weight
        user_embeds = self.user_embedding.weight

        # 原始的extended_id_embeds（用作RF的目标，向后兼容）
        extended_id_embeds_original = self.conv_ui(adj, user_embeds, item_embeds)

        # NEW: Generate h1 (historical) and h2 (augmented) if temporal mode enabled
        if self.use_temporal_augmentation and self.historical_adj is not None:
            h1_extended_id = self.conv_ui(self.historical_adj, user_embeds, item_embeds)
            h2_extended_id = self.conv_ui(self.augmented_adj, user_embeds, item_embeds)
        else:
            # Backward compatibility: no temporal mode
            h1_extended_id = torch.randn_like(extended_id_embeds_original)
            h2_extended_id = extended_id_embeds_original

        # ===== 显式的多模态特征（作为RF的条件） =====
        # Image modality
        explicit_image_item = self.conv_ii(self.image_original_adj, image_item_embeds)
        explicit_image_user = torch.sparse.mm(self.R, explicit_image_item)
        explicit_image_embeds = torch.cat(
            [explicit_image_user, explicit_image_item], dim=0
        )

        # Text modality
        explicit_text_item = self.conv_ii(self.text_original_adj, text_item_embeds)
        explicit_text_user = torch.sparse.mm(self.R, explicit_text_item)
        explicit_text_embeds = torch.cat(
            [explicit_text_user, explicit_text_item], dim=0
        )

        # ===== 使用RF生成extended_id_embeds =====
        rf_active = self.is_rf_active()

        if rf_active and train:
            # ===== RF独立训练：计算损失并反向传播 =====
            # 使用detach的条件，避免RF梯度影响GUME主模型
            image_cond_detached = explicit_image_embeds.detach()
            text_cond_detached = explicit_text_embeds.detach()
            h1_detached = h1_extended_id.detach()
            h2_detached = h2_extended_id.detach()

            # Check if using reflow (2-RF)
            use_reflow = (self.rf_reflow_mode == '2-step' and
                          self.current_epoch >= self.rf_reflow_epoch)

            if use_reflow:
                # Stage 2: Generate new X0 from trained 1-RF
                with torch.no_grad():
                    h1_reflow = self.rf_generator(
                        h1_detached,
                        image_cond_detached,
                        text_cond_detached,
                        n_steps=self.rf_sampling_steps
                    )
                # Train 2-RF on straighter paths
                rf_loss = self.rf_generator.rectified_flow_loss(
                    h1_reflow, h2_detached, image_cond_detached, text_cond_detached
                )
            else:
                # Stage 1: Normal RF training (h1 -> h2)
                rf_loss = self.rf_generator.rectified_flow_loss(
                    h1_detached, h2_detached, image_cond_detached, text_cond_detached
                )

            # Get current sampling steps
            current_steps = self.get_rf_sampling_steps()

            # 生成RF embeddings
            rf_generated_embeds = self.rf_generator(
                h1_detached, image_cond_detached, text_cond_detached,
                n_steps=current_steps
            )

            # Contrastive loss (ENABLE this - currently commented out at line 420)
            cl_loss = self.InfoNCE(rf_generated_embeds, h2_detached, self.rf_contrast_temp)

            # Total RF loss = velocity + weighted contrastive
            total_rf_loss = rf_loss + self.rf_loss_weight * cl_loss

            # RF独立反向传播和参数更新
            self.rf_optimizer.zero_grad()
            total_rf_loss.backward()
            self.rf_optimizer.step()

            # 打印RF训练信息（每个epoch只打印一次）
            if not self._rf_logged_this_epoch:
                mode_str = "2-RF" if use_reflow else "1-RF"
                print(
                    f"  [RF Train] epoch={self.current_epoch}, mode={mode_str}, "
                    f"steps={current_steps}, rf_loss={rf_loss.item():.6f}, "
                    f"cl_loss={cl_loss.item():.6f}, total={total_rf_loss.item():.6f}"
                )
                self._rf_logged_this_epoch = True

            # 混合模式：warmup期间使用纯GUME，之后进行混合
            if self.current_epoch < self.rf_warmup_epochs:
                # warmup阶段：RF在训练但不参与混合
                extended_id_embeds = h2_extended_id
            else:
                # warmup结束后：结合原始GUME和RF生成的embeddings
                extended_id_embeds = (
                    (1 - self.rf_mix_ratio) * h2_extended_id
                    + self.rf_mix_ratio * rf_generated_embeds.detach()
                )

        elif rf_active and not train:
            # 推理模式：使用混合比例
            current_steps = self.get_rf_sampling_steps()
            with torch.no_grad():
                if self.current_epoch < self.rf_warmup_epochs:
                    # warmup阶段：RF在训练但不参与混合
                    extended_id_embeds = h2_extended_id
                else:
                    rf_generated_embeds = self.rf_generator(
                        h1_extended_id,
                        explicit_image_embeds,
                        explicit_text_embeds,
                        n_steps=current_steps,
                    )
                    extended_id_embeds = (
                        (1 - self.rf_inference_mix_ratio) * h2_extended_id
                        + self.rf_inference_mix_ratio * rf_generated_embeds
                    )
        else:
            # 不使用RF 或 warmup阶段，保持原始GUME行为
            extended_id_embeds = h2_extended_id

        # ===== 继续GUME的其余部分 =====
        extended_image_embeds = self.conv_ui(
            adj, self.extended_image_user.weight, explicit_image_item
        )
        extended_text_embeds = self.conv_ui(
            adj, self.extended_text_user.weight, explicit_text_item
        )
        extended_it_embeds = (extended_image_embeds + extended_text_embeds) / 2

        # Attributes Separation for Better Integration
        image_weights, text_weights = torch.split(
            self.softmax(
                torch.cat(
                    [
                        self.separate_coarse(explicit_image_embeds),
                        self.separate_coarse(explicit_text_embeds),
                    ],
                    dim=-1,
                )
            ),
            1,
            dim=-1,
        )
        coarse_grained_embeds = (
            image_weights * explicit_image_embeds + text_weights * explicit_text_embeds
        )

        fine_grained_image = torch.multiply(
            self.image_behavior(extended_id_embeds),
            (explicit_image_embeds - coarse_grained_embeds),
        )
        fine_grained_text = torch.multiply(
            self.text_behavior(extended_id_embeds),
            (explicit_text_embeds - coarse_grained_embeds),
        )
        integration_embeds = (
            fine_grained_image + fine_grained_text + coarse_grained_embeds
        ) / 3

        all_embeds = extended_id_embeds + integration_embeds

        if train and rf_active:
            other_outputs = {
                "integration_embeds": integration_embeds,
                "extended_id_embeds": extended_id_embeds,
                "extended_it_embeds": extended_it_embeds,
                "explicit_image_embeds": explicit_image_embeds,
                "explicit_text_embeds": explicit_text_embeds,
            }

            return all_embeds, other_outputs
        elif train and not rf_active:
            return (
                all_embeds,
                (integration_embeds, extended_id_embeds, extended_it_embeds),
                (explicit_image_embeds, explicit_text_embeds),
            )

        return all_embeds

    def calculate_loss(self, interaction):
        """
        计算总损失（包括原始GUME损失和RF损失）

        Args:
            interaction: 交互数据

        Returns:
            total_loss: 总损失
        """
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        # 前向传播（RF损失已在forward中独立计算和反向传播）
        if self.is_rf_active():
            embeds_1, other_outputs = self.forward(self.norm_adj, train=True)

            integration_embeds = other_outputs["integration_embeds"]
            extended_id_embeds = other_outputs["extended_id_embeds"]
            extended_it_embeds = other_outputs["extended_it_embeds"]
            explicit_image_embeds = other_outputs["explicit_image_embeds"]
            explicit_text_embeds = other_outputs["explicit_text_embeds"]
        else:
            embeds_1, embeds_2, embeds_3 = self.forward(self.norm_adj, train=True)
            integration_embeds, extended_id_embeds, extended_it_embeds = embeds_2
            explicit_image_embeds, explicit_text_embeds = embeds_3

        users_embeddings, items_embeddings = torch.split(
            embeds_1, [self.n_users, self.n_items], dim=0
        )

        # ===== 原始GUME损失 =====
        u_g_embeddings = users_embeddings[users]
        pos_i_g_embeddings = items_embeddings[pos_items]
        neg_i_g_embeddings = items_embeddings[neg_items]

        vt_loss = self.vt_loss * self.align_vt(
            explicit_image_embeds, explicit_text_embeds
        )

        integration_users, integration_items = torch.split(
            integration_embeds, [self.n_users, self.n_items], dim=0
        )
        extended_id_user, extended_id_items = torch.split(
            extended_id_embeds, [self.n_users, self.n_items], dim=0
        )
        bpr_loss, reg_loss_1 = self.bpr_loss(
            u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings
        )

        bm_loss = self.bm_loss * (
            self.InfoNCE(
                integration_users[users], extended_id_user[users], self.bm_temp
            )
            + self.InfoNCE(
                integration_items[pos_items], extended_id_items[pos_items], self.bm_temp
            )
        )

        al_loss = vt_loss + bm_loss

        extended_it_user, extended_it_items = torch.split(
            extended_it_embeds, [self.n_users, self.n_items], dim=0
        )

        c_loss = self.InfoNCE(
            extended_it_user[users], integration_users[users], self.um_temp
        )
        noise_loss_1 = self.cal_noise_loss(users, integration_users, self.um_temp)
        noise_loss_2 = self.cal_noise_loss(users, extended_it_user, self.um_temp)
        um_loss = self.um_loss * (c_loss + noise_loss_1 + noise_loss_2)

        reg_loss_2 = (
            self.reg_weight_2
            * self.sq_sum(extended_it_items[pos_items])
            / self.batch_size
        )
        reg_loss = reg_loss_1 + reg_loss_2

        # 返回GUME损失（RF损失已在forward中独立处理）
        return bpr_loss + al_loss + um_loss + reg_loss

    def full_sort_predict(self, interaction):
        """
        预测（推理模式）
        """
        user = interaction[0]

        if self.use_rf:
            all_embeds = self.forward(self.norm_adj, train=False)
        else:
            all_embeds = self.forward(self.norm_adj)

        restore_user_e, restore_item_e = torch.split(
            all_embeds, [self.n_users, self.n_items], dim=0
        )
        u_embeddings = restore_user_e[user]

        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores

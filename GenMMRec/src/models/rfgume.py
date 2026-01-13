# Rectified Flow for GUME: RF-GUME
# Integrating Rectified Flow into GUME for generating extended_id_embeds

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.gume import GUME


def cosine_similarity_gradient(x_t, x_1):
    """
    计算 cos_sim(x_t, x_1) 相对于 x_t 的梯度

    梯度公式: ∇_{x_t} cos_sim(x_t, x_1) = (x_1 / ||x_1||) / ||x_t|| - (x_t / ||x_t||) * cos_sim / ||x_t||

    Args:
        x_t: 当前状态, shape (batch, embedding_dim)
        x_1: 目标状态, shape (batch, embedding_dim)

    Returns:
        grad: 余弦相似度的梯度, shape (batch, embedding_dim)
    """
    # 计算余弦相似度
    cos_sim = F.cosine_similarity(x_t, x_1, dim=-1)  # (batch,)
    cos_sim = cos_sim.unsqueeze(-1)  # (batch, 1)

    # 计算范数
    x_t_norm = x_t.norm(dim=-1, keepdim=True)  # (batch, 1)
    x_t_norm = torch.clamp(x_t_norm, min=1e-8)  # 避免除零

    # 归一化向量
    x_1_normalized = F.normalize(x_1, dim=-1)  # (batch, embedding_dim)
    x_t_normalized = F.normalize(x_t, dim=-1)  # (batch, embedding_dim)

    # 计算梯度
    grad = x_1_normalized / x_t_norm - x_t_normalized * cos_sim / x_t_norm

    return grad


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

    def __init__(self, embedding_dim, hidden_dim, n_layers, dropout, condition_dim,
                 user_guidance_scale=0.2, guidance_decay_power=2.0,
                 cosine_guidance_scale=0.1, cosine_decay_power=2.0):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.user_guidance_scale = user_guidance_scale
        self.guidance_decay_power = guidance_decay_power
        self.cosine_guidance_scale = cosine_guidance_scale
        self.cosine_decay_power = cosine_decay_power

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

    def forward(self, x, t, conditions, user_prior=None, x_1=None):
        """
        Args:
            x: 当前状态 X_t, shape (batch, embedding_dim)
            t: 时间步, shape (batch, 1)
            conditions: 条件特征（image + text）, shape (batch, condition_dim)
            user_prior: 用户特定兴趣指导, shape (batch, embedding_dim)
            x_1: 目标状态 X_1, shape (batch, embedding_dim) - 用于计算余弦梯度

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

        # [NEW] 添加先验知识指导（仅训练模式）
        if self.training:
            # 第一项: 用户兴趣先验指导
            if user_prior is not None:
                # 时间衰减权重: lambda_1(t) = (1-t)^power
                lambda_1 = (1 - t) ** self.guidance_decay_power  # shape: (batch, 1)

                # 添加用户先验指导项
                v = v + lambda_1 * self.user_guidance_scale * user_prior

            # 第二项: 余弦相似度梯度指导
            if x_1 is not None:
                # 时间衰减权重: lambda_2(t) = (1-t)^power
                lambda_2 = (1 - t) ** self.cosine_decay_power  # shape: (batch, 1)

                # 计算余弦相似度梯度
                cos_grad = cosine_similarity_gradient(x, x_1)

                # 添加余弦梯度指导项
                v = v + lambda_2 * self.cosine_guidance_scale * cos_grad

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

    def __init__(self, embedding_dim, hidden_dim, n_layers, dropout,
                 user_guidance_scale=0.2, guidance_decay_power=2.0,
                 cosine_guidance_scale=0.1, cosine_decay_power=2.0):
        super().__init__()

        self.embedding_dim = embedding_dim

        # 速度场网络（条件维度 = embedding_dim * 2，因为有image和text两个条件）
        self.velocity_net = SimpleVelocityNet(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            condition_dim=embedding_dim * 2,  # image + text conditions
            user_guidance_scale=user_guidance_scale,
            guidance_decay_power=guidance_decay_power,
            cosine_guidance_scale=cosine_guidance_scale,
            cosine_decay_power=cosine_decay_power,
        )

    def rectified_flow_loss(self, target_embeds, image_cond, text_cond, user_prior=None):
        """
        计算Rectified Flow损失（带有可选的用户兴趣先验指导）

        Loss = E_t [||v(X_t, t, c) - (X1 - X0)||^2]
        其中 X_t = t*X1 + (1-t)*X0

        Args:
            target_embeds: 目标的extended_id_embeds (X1), shape (batch, embedding_dim)
            image_cond: explicit_image_item 条件, shape (batch, embedding_dim)
            text_cond: explicit_text_item 条件, shape (batch, embedding_dim)
            user_prior: 用户兴趣先验指导 (Z_u - Z_hat), shape (batch, embedding_dim) [NEW]

        Returns:
            rf_loss: Rectified Flow损失
        """
        batch_size = target_embeds.shape[0]

        # X1 = 目标嵌入
        X1 = target_embeds

        # X0 = 随机噪声
        X0 = torch.randn_like(X1)

        # 随机采样时间步 t ~ Uniform[0, 1]
        t = torch.rand(batch_size, 1).to(X1.device)

        # 线性插值: X_t = t*X1 + (1-t)*X0
        X_t = t * X1 + (1 - t) * X0

        # 拼接条件（image + text）
        conditions = torch.cat([image_cond, text_cond], dim=-1)

        # [MODIFIED] 预测速度: v(X_t, t, conditions, user_prior, X1)
        v_pred = self.velocity_net(X_t, t, conditions, user_prior=user_prior, x_1=X1)

        # 目标速度: X1 - X0（直线方向）
        v_target = X1 - X0

        # Rectified Flow损失（MSE）
        rf_loss = F.mse_loss(v_pred, v_target)

        return rf_loss

    def sample_ode(self, z_0, image_cond, text_cond, n_steps=10):
        """
        使用Euler方法求解ODE生成extended_id_embeds

        dZ_t = v(Z_t, t, conditions) dt

        Args:
            z_0: 初始噪声, shape (batch, embedding_dim)
            image_cond: explicit_image_item 条件
            text_cond: explicit_text_item 条件
            n_steps: ODE求解步数

        Returns:
            z_1: 生成的extended_id_embeds
        """
        z_t = z_0
        dt = 1.0 / n_steps

        # 拼接条件
        conditions = torch.cat([image_cond, text_cond], dim=-1)

        # Euler方法求解ODE
        for i in range(n_steps):
            t = torch.full((z_0.shape[0], 1), i * dt).to(z_0.device)
            v = self.velocity_net(z_t, t, conditions)
            z_t = z_t + v * dt

        return z_t

    def forward(self, image_cond, text_cond, n_steps=10):
        """
        前向传播：生成extended_id_embeds

        Args:
            image_cond: explicit_image_item 条件
            text_cond: explicit_text_item 条件
            n_steps: ODE求解步数

        Returns:
            generated_embeds: 生成的extended_id_embeds
        """
        batch_size = image_cond.shape[0]

        # 从标准高斯噪声开始
        z_0 = torch.randn(batch_size, self.embedding_dim).to(image_cond.device)

        # 求解ODE生成嵌入
        generated_embeds = self.sample_ode(z_0, image_cond, text_cond, n_steps)

        return generated_embeds


class RFGUME(GUME):
    """
    GUME with Rectified Flow

    在GUME的基础上，使用Rectified Flow生成extended_id_embeds
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

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

        # 对比损失权重参数
        self.rf_loss_weight = (
            config["rf_loss_weight"] if "rf_loss_weight" in config else 1.0
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

        # 用户先验指导参数
        self.use_user_guidance = (
            config["use_user_guidance"] if "use_user_guidance" in config else True
        )
        self.user_guidance_scale = (
            config["user_guidance_scale"] if "user_guidance_scale" in config else 0.2
        )
        self.guidance_decay_power = (
            config["guidance_decay_power"] if "guidance_decay_power" in config else 2.0
        )

        # 余弦相似度梯度指导参数
        self.use_cosine_guidance = (
            config["use_cosine_guidance"] if "use_cosine_guidance" in config else True
        )
        self.cosine_guidance_scale = (
            config["cosine_guidance_scale"] if "cosine_guidance_scale" in config else 0.1
        )
        self.cosine_decay_power = (
            config["cosine_decay_power"] if "cosine_decay_power" in config else 2.0
        )

        if self.use_rf:
            # 使用单个统一的RF生成器处理用户和物品
            self.rf_generator = RFExtendedIdGenerator(
                embedding_dim=self.embedding_dim,
                hidden_dim=self.rf_hidden_dim,
                n_layers=self.rf_n_layers,
                dropout=self.rf_dropout,
                user_guidance_scale=self.user_guidance_scale,
                guidance_decay_power=self.guidance_decay_power,
                cosine_guidance_scale=self.cosine_guidance_scale,
                cosine_decay_power=self.cosine_decay_power,
            )

            # RF独立优化器
            self.rf_optimizer = torch.optim.AdamW(
                self.rf_generator.parameters(), lr=self.rf_learning_rate
            )

            # 用于控制每个epoch只打印一次日志
            self._rf_logged_this_epoch = False

            # 用于存储上一个epoch生成的embeddings（用于余弦梯度计算）
            # 初始化为 None，第一次训练时会被初始化
            self.prev_generated_embeds = None

            print(
                f"RF-GUME initialized: "
                f"hidden_dim={self.rf_hidden_dim}, n_layers={self.rf_n_layers}, "
                f"sampling_steps={self.rf_sampling_steps}, "
                f"warmup_epochs={self.rf_warmup_epochs}, "
                f"rf_lr={self.rf_learning_rate}, "
                f"mix_ratio(train/infer)={self.rf_mix_ratio}/{self.rf_inference_mix_ratio}\n"
                f"  User guidance: enabled={self.use_user_guidance}, "
                f"scale={self.user_guidance_scale}, decay={self.guidance_decay_power}\n"
                f"  Cosine guidance: enabled={self.use_cosine_guidance}, "
                f"scale={self.cosine_guidance_scale}, decay={self.cosine_decay_power}"
            )

    def set_epoch(self, epoch):
        """由 trainer 调用，更新当前 epoch"""
        self.current_epoch = epoch
        if self.use_rf:
            self._rf_logged_this_epoch = False  # 重置日志标记

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

        # 原始的extended_id_embeds（用作RF的目标）
        extended_id_embeds_target = self.conv_ui(adj, user_embeds, item_embeds)

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
            target_detached = extended_id_embeds_target.detach()

            # [NEW] 计算用户兴趣先验
            # Z_u: 用户特定的多模态兴趣表示
            Z_u = explicit_image_embeds[:self.n_users] + explicit_text_embeds[:self.n_users]

            # Z_hat: 通用兴趣表示（所有用户的平均值）
            Z_hat = Z_u.mean(dim=0, keepdim=True)

            # 用户先验: 独特的用户兴趣
            user_prior = Z_u - Z_hat  # shape: (n_users, embedding_dim)

            # 对于物品，不使用个性化指导（零指导）
            item_prior = torch.zeros(self.n_items, self.embedding_dim).to(Z_u.device)

            # 合并用户和物品先验
            full_prior = torch.cat([user_prior, item_prior], dim=0)  # shape: (n_users+n_items, embedding_dim)

            # Detach 用于 RF 训练
            full_prior_detached = full_prior.detach()

            # [MODIFIED] 计算RF速度场损失（带有用户先验）
            rf_loss = self.rf_generator.rectified_flow_loss(
                target_detached,
                image_cond_detached,
                text_cond_detached,
                user_prior=full_prior_detached,
            )

            # 生成RF embeddings
            rf_generated_embeds = self.rf_generator(
                image_cond_detached,
                text_cond_detached,
                n_steps=self.rf_sampling_steps,
            )

            # 对比损失：约束RF生成的向量接近原始目标向量
            cl_loss = self.InfoNCE(rf_generated_embeds, target_detached, self.rf_contrast_temp)

            # 总RF损失 = 速度场损失 + 加权对比约束
            total_rf_loss = rf_loss + self.rf_loss_weight * cl_loss

            # RF独立反向传播和参数更新
            self.rf_optimizer.zero_grad()
            total_rf_loss.backward()
            self.rf_optimizer.step()

            # 打印RF训练信息（每个epoch只打印一次）
            if not self._rf_logged_this_epoch:
                print(
                    f"  [RF Train] epoch={self.current_epoch}, "
                    f"rf_loss={rf_loss.item():.6f}, cl_loss={cl_loss.item():.6f}"
                )
                self._rf_logged_this_epoch = True

            # 混合模式：warmup期间使用纯GUME，之后进行混合
            if self.current_epoch < self.rf_warmup_epochs:
                # warmup阶段：RF在训练但不参与混合
                extended_id_embeds = extended_id_embeds_target
            else:
                # warmup结束后：结合原始GUME和RF生成的embeddings
                extended_id_embeds = (
                    (1 - self.rf_mix_ratio) * extended_id_embeds_target
                    + self.rf_mix_ratio * rf_generated_embeds.detach()
                )

        elif rf_active and not train:
            # 推理模式：使用混合比例
            with torch.no_grad():
                if self.current_epoch < self.rf_warmup_epochs:
                    # warmup阶段：RF在训练但不参与混合
                    extended_id_embeds = extended_id_embeds_target
                else:
                    rf_generated_embeds = self.rf_generator(
                        explicit_image_embeds,
                        explicit_text_embeds,
                        n_steps=self.rf_sampling_steps,
                    )
                    extended_id_embeds = (
                        (1 - self.rf_inference_mix_ratio) * extended_id_embeds_target
                        + self.rf_inference_mix_ratio * rf_generated_embeds
                    )
        else:
            # 不使用RF 或 warmup阶段，保持原始GUME行为
            extended_id_embeds = extended_id_embeds_target

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

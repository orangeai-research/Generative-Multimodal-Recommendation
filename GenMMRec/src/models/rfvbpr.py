# coding: utf-8
# @email: enoche.chow@gmail.com
r"""
RFVBPR: RF-Enhanced VBPR
Integrates Rectified Flow module to enhance visual-aware recommendations
With optional causal denoising using Inverse Propensity Weighting (IPW)
"""

import torch
import torch.nn.functional as F

from models.vbpr import VBPR
from models.rf_modules import RFEmbeddingGenerator, CausalDenoiser


class RFVBPR(VBPR):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.use_rf = config["use_rf"] if "use_rf" in config else True

        if self.use_rf:
            # Initialize RF generator
            self.rf_generator = RFEmbeddingGenerator(
                embedding_dim=self.i_embedding_size * 2,  # VBPR uses 2x embedding size
                hidden_dim=config["rf_hidden_dim"] if "rf_hidden_dim" in config else 128,
                n_layers=config["rf_n_layers"] if "rf_n_layers" in config else 2,
                dropout=config["rf_dropout"] if "rf_dropout" in config else 0.1,
                learning_rate=config["rf_learning_rate"] if "rf_learning_rate" in config else 0.0001,
                sampling_steps=config["rf_sampling_steps"] if "rf_sampling_steps" in config else 10,
                warmup_epochs=config["rf_warmup_epochs"] if "rf_warmup_epochs" in config else 5,
                train_mix_ratio=config["rf_mix_ratio"] if "rf_mix_ratio" in config else 0.1,
                inference_mix_ratio=config["rf_inference_mix_ratio"] if "rf_inference_mix_ratio" in config else 0.2,
                contrast_temp=config["rf_contrast_temp"] if "rf_contrast_temp" in config else 0.2,
                contrast_weight=config["rf_loss_weight"] if "rf_loss_weight" in config else 1.0,
                n_users=self.n_users,
                n_items=self.n_items,
                use_2rf=config["use_2rf"] if "use_2rf" in config else False,
                rf_2rf_transition_epoch=config["rf_2rf_transition_epoch"] if "rf_2rf_transition_epoch" in config else None,
                use_gradient_checkpointing=config["use_gradient_checkpointing"] if "use_gradient_checkpointing" in config else True,
            )
            self._rf_logged_this_epoch = False
            self._current_batch_users = None
            self._current_batch_items = None
            self._training_epoch = -1

        # Denoising Module
        self.use_denoise = config["use_denoise"] if "use_denoise" in config else False

        if self.use_denoise:
            self.ps_loss_weight = config["ps_loss_weight"] if "ps_loss_weight" in config else 0.1
            self.causal_denoiser = CausalDenoiser(
                embedding_dim=self.i_embedding_size * 2,
                n_users=self.n_users,
                n_items=self.n_items,
                n_layers=config["denoise_layers"] if "denoise_layers" in config else 2,
                clean_rating_threshold=config["clean_rating_threshold"] if "clean_rating_threshold" in config else 5.0,
                device=self.device,
            )
            self.causal_denoiser.load_treatment_labels(dataset)

    def pre_epoch_processing(self):
        """Called by trainer at the beginning of each epoch."""
        if self.use_rf:
            self._training_epoch += 1
            self.rf_generator.set_epoch(self._training_epoch)
            self._rf_logged_this_epoch = False

    def forward(self, dropout=0.0):
        # Get item embeddings with visual features
        item_embeddings = self.item_linear(self.item_raw_features)
        item_embeddings = torch.cat((self.i_embedding, item_embeddings), -1)

        user_e = F.dropout(self.u_embedding, dropout)
        item_e = F.dropout(item_embeddings, dropout)

        # Combine user and item embeddings
        all_embeddings_ori = torch.cat([user_e, item_e], dim=0)

        # ===== RF Enhancement =====
        rf_outputs = None

        if self.use_rf and self.training:
            print(f"[RFVBPR] Forward in TRAINING mode")
            ps_loss = 0.0
            if self.use_denoise:
                # For VBPR, use combined user-item embeddings for denoising
                ego_emb_for_denoise = torch.cat([self.u_embedding, item_embeddings], dim=0)
                denoised_emb, ps_loss = self.causal_denoiser(ego_emb_for_denoise)
                if denoised_emb is not None:
                    rf_target = denoised_emb.detach()
                else:
                    rf_target = all_embeddings_ori.detach()
            else:
                rf_target = all_embeddings_ori.detach()

            # Prepare multimodal conditions
            conditions = []
            if self.item_raw_features is not None:
                # Use raw visual/text features as conditions
                # Extend to user+item space (users get aggregated item features)
                item_feat_transformed = self.item_linear(self.item_raw_features)
                # For users, aggregate item features via interaction (simplified: use zeros)
                user_feat = torch.zeros(self.n_users, item_feat_transformed.shape[1]).to(item_feat_transformed.device)
                full_feat = torch.cat([user_feat, item_feat_transformed], dim=0)
                conditions.append(full_feat.detach())

            # Compute user prior (personalized visual preferences)
            # Use the actual embeddings to compute prior, not just the visual features
            user_embeds_for_prior = all_embeddings_ori[:self.n_users]
            item_embeds_for_prior = all_embeddings_ori[self.n_users:]
            
            # User prior: deviation from average
            avg_user_embed = user_embeds_for_prior.mean(dim=0, keepdim=True)
            user_prior = user_embeds_for_prior - avg_user_embed
            
            # Item prior: deviation from average
            avg_item_embed = item_embeds_for_prior.mean(dim=0, keepdim=True)
            item_prior = item_embeds_for_prior - avg_item_embed
            
            full_prior = torch.cat([user_prior, item_prior], dim=0)

            loss_dict = self.rf_generator.compute_loss_and_step(
                target_embeds=rf_target,
                conditions=conditions,
                user_prior=full_prior,
                epoch=self.rf_generator.current_epoch,
                batch_users=self._current_batch_users,
                batch_pos_items=self._current_batch_items,
            )

            if not self._rf_logged_this_epoch:
                print(
                    f"  [RF Train] epoch={self.rf_generator.current_epoch}, "
                    f"rf_loss={loss_dict['rf_loss']:.6f}, "
                    f"cl_loss={loss_dict['cl_loss']:.6f}"
                )
                self._rf_logged_this_epoch = True

            # Generate RF embeddings
            rf_embeds = self.rf_generator.generate(conditions)

            # Mix original and RF generated embeddings
            all_embeddings = self.rf_generator.mix_embeddings(
                all_embeddings_ori,
                rf_embeds.detach(),
                training=True,
                epoch=self.rf_generator.current_epoch,
            )

            rf_outputs = {"ps_loss": ps_loss}
        elif self.use_rf and not self.training:
            print(f"[RFVBPR] Forward in INFERENCE mode")
            # Inference mode
            with torch.no_grad():
                # Prepare multimodal conditions
                conditions = []
                if self.item_raw_features is not None:
                    item_feat_transformed = self.item_linear(self.item_raw_features)
                    user_feat = torch.zeros(self.n_users, item_feat_transformed.shape[1]).to(item_feat_transformed.device)
                    full_feat = torch.cat([user_feat, item_feat_transformed], dim=0)
                    conditions.append(full_feat)
                
                rf_embeds = self.rf_generator.generate(conditions)
                all_embeddings = self.rf_generator.mix_embeddings(
                    all_embeddings_ori,
                    rf_embeds,
                    training=False,
                    epoch=self.rf_generator.current_epoch,
                )
        else:
            all_embeddings = all_embeddings_ori

        user_e, item_e = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)

        if self.use_rf and self.training:
            return user_e, item_e, rf_outputs
        return user_e, item_e

    def calculate_loss(self, interaction):
        # Store batch indices for RF contrastive loss
        self._current_batch_users = interaction[0]
        self._current_batch_items = interaction[1]

        user = interaction[0]
        pos_item = interaction[1]
        neg_item = interaction[2]

        if self.use_rf:
            user_embeddings, item_embeddings, rf_outputs = self.forward()
        else:
            user_embeddings, item_embeddings = self.forward()
            rf_outputs = None

        user_e = user_embeddings[user, :]
        pos_e = item_embeddings[pos_item, :]
        neg_e = item_embeddings[neg_item, :]

        pos_item_score = torch.mul(user_e, pos_e).sum(dim=1)
        neg_item_score = torch.mul(user_e, neg_e).sum(dim=1)
        mf_loss = self.loss(pos_item_score, neg_item_score)
        reg_loss = self.reg_loss(user_e, pos_e, neg_e)
        total_loss = mf_loss + self.reg_weight * reg_loss

        # Add propensity score loss if denoising is enabled
        if self.use_denoise and rf_outputs is not None and "ps_loss" in rf_outputs:
            total_loss = total_loss + self.ps_loss_weight * rf_outputs["ps_loss"]

        return total_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]

        with torch.no_grad():
            if self.use_rf:
                # Get item embeddings with visual features
                item_embeddings = self.item_linear(self.item_raw_features)
                item_embeddings = torch.cat((self.i_embedding, item_embeddings), -1)
                all_embeddings_ori = torch.cat([self.u_embedding, item_embeddings], dim=0)

                # Prepare conditions
                conditions = []
                if self.item_raw_features is not None:
                    item_feat_transformed = self.item_linear(self.item_raw_features)
                    user_feat = torch.zeros(self.n_users, item_feat_transformed.shape[1]).to(item_feat_transformed.device)
                    full_feat = torch.cat([user_feat, item_feat_transformed], dim=0)
                    conditions.append(full_feat)

                # Generate RF embeddings
                rf_embeds = self.rf_generator.generate(conditions)
                all_embeddings = self.rf_generator.mix_embeddings(
                    all_embeddings_ori,
                    rf_embeds,
                    training=False,
                    epoch=self.rf_generator.current_epoch,
                )

                user_embeddings, item_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
            else:
                user_embeddings, item_embeddings = self.forward()

        user_e = user_embeddings[user, :]
        all_item_e = item_embeddings
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score

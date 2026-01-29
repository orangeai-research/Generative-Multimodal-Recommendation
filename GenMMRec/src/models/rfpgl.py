# coding: utf-8
# @email: enoche.chow@gmail.com
r"""
RFPGL: RF-Enhanced PGL
Integrates Rectified Flow module to enhance principal graph learning embeddings
With optional causal denoising using Inverse Propensity Weighting (IPW)
"""

import torch
import torch.nn.functional as F

from models.pgl import PGL
from models.rf_modules import RFEmbeddingGenerator, CausalDenoiser


class RFPGL(PGL):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.use_rf = config["use_rf"] if "use_rf" in config else True
        # PGL uses concatenated embeddings (image + text)
        self.total_embedding_dim = self.embedding_dim * 2  # user_image + user_text or image_feats + text_feats

        if self.use_rf:
            # Initialize RF generator
            self.rf_generator = RFEmbeddingGenerator(
                embedding_dim=self.total_embedding_dim,
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
                embedding_dim=self.total_embedding_dim,
                n_users=self.n_users,
                n_items=self.n_items,
                n_layers=config["denoise_layers"] if "denoise_layers" in config else 2,
                clean_rating_threshold=config["clean_rating_threshold"] if "clean_rating_threshold" in config else 5.0,
                device=self.device,
            )
            self.causal_denoiser.load_treatment_labels(dataset)

    def pre_epoch_processing(self):
        """Called by trainer at the beginning of each epoch."""
        super().pre_epoch_processing()
        
        if self.use_rf:
            self._training_epoch += 1
            self.rf_generator.set_epoch(self._training_epoch)
            self._rf_logged_this_epoch = False

    def forward(self, adj, train=False):
        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
        if self.t_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight)

        image_feats, text_feats = F.normalize(image_feats), F.normalize(text_feats)
        user_embeds = torch.cat([self.user_image.weight, self.user_text.weight], dim=1)
        item_embeds = torch.cat([image_feats, text_feats], dim=1)

        h = item_embeds
        for i in range(self.n_layers):
            h = torch.sparse.mm(self.mm_adj, h)

        ego_embeddings = torch.cat((user_embeds, item_embeds), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        i_g_embeddings_final = i_g_embeddings + h

        # ===== RF Enhancement =====
        rf_outputs = None

        if self.use_rf and self.training:
            # print(f"[RFPGL] Forward in TRAINING mode")
            # Combine user and item embeddings
            all_embeds = torch.cat([u_g_embeddings, i_g_embeddings_final], dim=0)
            
            ps_loss = 0.0
            if self.use_denoise:
                denoised_emb, ps_loss = self.causal_denoiser(ego_embeddings)
                if denoised_emb is not None:
                    rf_target = denoised_emb.detach()
                else:
                    rf_target = all_embeds.detach()
            else:
                rf_target = all_embeds.detach()

            # Prepare multimodal conditions
            conditions = []
            
            # Visual features (transformed)
            if self.v_feat is not None:
                # Extend to user+item space
                user_v_feat = torch.zeros(self.n_users, image_feats.shape[1]).to(image_feats.device)
                full_v_feat = torch.cat([user_v_feat, image_feats], dim=0)
                conditions.append(full_v_feat.detach())
            
            # Text features (transformed)
            if self.t_feat is not None:
                # Extend to user+item space
                user_t_feat = torch.zeros(self.n_users, text_feats.shape[1]).to(text_feats.device)
                full_t_feat = torch.cat([user_t_feat, text_feats], dim=0)
                conditions.append(full_t_feat.detach())

            # Compute user prior (principal graph deviation)
            # Use the actual embeddings to compute prior, not the conditions
            user_embeds_for_prior = all_embeds[:self.n_users]
            item_embeds_for_prior = all_embeds[self.n_users:]
            
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
            all_embeds = self.rf_generator.mix_embeddings(
                all_embeds,
                rf_embeds.detach(),
                training=True,
                epoch=self.rf_generator.current_epoch,
            )

            u_g_embeddings, i_g_embeddings_final = torch.split(all_embeds, [self.n_users, self.n_items], dim=0)

            rf_outputs = {"ps_loss": ps_loss}
        elif self.use_rf and not self.training:
            # print(f"[RFPGL] Forward in INFERENCE mode")
            # Inference mode
            with torch.no_grad():
                # Combine user and item embeddings
                all_embeds = torch.cat([u_g_embeddings, i_g_embeddings_final], dim=0)
                
                # Prepare multimodal conditions
                conditions = []
                if self.v_feat is not None:
                    user_v_feat = torch.zeros(self.n_users, image_feats.shape[1]).to(image_feats.device)
                    full_v_feat = torch.cat([user_v_feat, image_feats], dim=0)
                    conditions.append(full_v_feat)
                if self.t_feat is not None:
                    user_t_feat = torch.zeros(self.n_users, text_feats.shape[1]).to(text_feats.device)
                    full_t_feat = torch.cat([user_t_feat, text_feats], dim=0)
                    conditions.append(full_t_feat)
                
                rf_embeds = self.rf_generator.generate(conditions)
                all_embeds = self.rf_generator.mix_embeddings(
                    all_embeds,
                    rf_embeds,
                    training=False,
                    epoch=self.rf_generator.current_epoch,
                )
                
                u_g_embeddings, i_g_embeddings_final = torch.split(all_embeds, [self.n_users, self.n_items], dim=0)

        if train and self.use_rf:
            return u_g_embeddings, i_g_embeddings_final, rf_outputs
        return u_g_embeddings, i_g_embeddings_final

    def calculate_loss(self, interaction):
        # Store batch indices for RF contrastive loss
        self._current_batch_users = interaction[0]
        self._current_batch_items = interaction[1]

        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        if self.use_rf:
            ua_embeddings, ia_embeddings, rf_outputs = self.forward(self.sub_graph, train=True)
        else:
            ua_embeddings, ia_embeddings = self.forward(self.sub_graph)
            rf_outputs = None

        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        batch_mf_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)
        cl_loss = (self.InfoNCE(self.dropoutf(u_g_embeddings), self.dropoutf(u_g_embeddings), 0.2)
                   + self.InfoNCE(self.dropoutf(pos_i_g_embeddings), self.dropoutf(pos_i_g_embeddings), 0.2)) / 2
        
        total_loss = batch_mf_loss + self.reg_weight * cl_loss

        # Add propensity score loss if denoising is enabled
        if self.use_denoise and rf_outputs is not None and "ps_loss" in rf_outputs:
            total_loss = total_loss + self.ps_loss_weight * rf_outputs["ps_loss"]

        return total_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]

        restore_user_e, restore_item_e = self.forward(self.norm_adj)
        u_embeddings = restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores

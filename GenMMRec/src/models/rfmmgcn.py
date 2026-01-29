# coding: utf-8
# @email: enoche.chow@gmail.com
r"""
RFMMGCN: RF-Enhanced MMGCN
Integrates Rectified Flow module to enhance multi-modal graph convolution embeddings
With optional causal denoising using Inverse Propensity Weighting (IPW)
"""

import torch
import torch.nn.functional as F

from models.mmgcn import MMGCN
from models.rf_modules import RFEmbeddingGenerator, CausalDenoiser


class RFMMGCN(MMGCN):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.use_rf = config["use_rf"] if "use_rf" in config else True
        self.dim_x = config['embedding_size']

        if self.use_rf:
            # Initialize RF generator
            self.rf_generator = RFEmbeddingGenerator(
                embedding_dim=self.dim_x,
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
                embedding_dim=self.dim_x,
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

    def forward(self):
        representation = None
        if self.v_feat is not None:
            representation = self.v_gcn(self.v_feat, self.id_embedding)
        if self.t_feat is not None:
            if representation is None:
                representation = self.t_gcn(self.t_feat, self.id_embedding)
            else:
                representation += self.t_gcn(self.t_feat, self.id_embedding)

        representation /= self.num_modal

        # ===== RF Enhancement =====
        rf_outputs = None

        if self.use_rf and self.training:
            print(f"[RFMMGCN] Forward in TRAINING mode")
            ps_loss = 0.0
            if self.use_denoise:
                ego_emb_for_denoise = self.id_embedding
                denoised_emb, ps_loss = self.causal_denoiser(ego_emb_for_denoise)
                if denoised_emb is not None:
                    rf_target = denoised_emb.detach()
                else:
                    rf_target = representation.detach()
            else:
                rf_target = representation.detach()

            # Prepare multimodal conditions
            conditions = []
            
            # Visual features
            if self.v_feat is not None:
                # Get visual GCN output as condition
                v_representation = self.v_gcn(self.v_feat, self.id_embedding)
                conditions.append(v_representation.detach())
            
            # Text features
            if self.t_feat is not None:
                # Get text GCN output as condition
                t_representation = self.t_gcn(self.t_feat, self.id_embedding)
                conditions.append(t_representation.detach())

            # Compute user prior (multimodal preference deviation)
            if len(conditions) > 0:
                # Combine all modality representations
                combined_modal = sum(conditions) / len(conditions)
                
                # Split user and item embeddings
                user_modal = combined_modal[:self.n_users]
                item_modal = combined_modal[self.n_users:]
                
                # User prior: deviation from average user preference
                avg_user_modal = user_modal.mean(dim=0, keepdim=True)
                user_prior = user_modal - avg_user_modal
                
                # Item prior: deviation from average item features
                avg_item_modal = item_modal.mean(dim=0, keepdim=True)
                item_prior = item_modal - avg_item_modal
                
                full_prior = torch.cat([user_prior, item_prior], dim=0)
            else:
                full_prior = None

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
            representation = self.rf_generator.mix_embeddings(
                representation,
                rf_embeds.detach(),
                training=True,
                epoch=self.rf_generator.current_epoch,
            )

            rf_outputs = {"ps_loss": ps_loss}
        elif self.use_rf and not self.training:
            print(f"[RFMMGCN] Forward in INFERENCE mode")
            # Inference mode
            with torch.no_grad():
                # Prepare multimodal conditions
                conditions = []
                if self.v_feat is not None:
                    v_representation = self.v_gcn(self.v_feat, self.id_embedding)
                    conditions.append(v_representation)
                if self.t_feat is not None:
                    t_representation = self.t_gcn(self.t_feat, self.id_embedding)
                    conditions.append(t_representation)
                
                rf_embeds = self.rf_generator.generate(conditions)
                representation = self.rf_generator.mix_embeddings(
                    representation,
                    rf_embeds,
                    training=False,
                    epoch=self.rf_generator.current_epoch,
                )

        self.result = representation

        if self.use_rf and self.training:
            return representation, rf_outputs
        return representation

    def calculate_loss(self, interaction):
        # Store batch indices for RF contrastive loss
        self._current_batch_users = interaction[0]
        self._current_batch_items = interaction[1]

        batch_users = interaction[0]
        pos_items = interaction[1] + self.n_users
        neg_items = interaction[2] + self.n_users

        user_tensor = batch_users.repeat_interleave(2)
        stacked_items = torch.stack((pos_items, neg_items))
        item_tensor = stacked_items.t().contiguous().view(-1)

        if self.use_rf:
            out, rf_outputs = self.forward()
        else:
            out = self.forward()
            rf_outputs = None

        user_score = out[user_tensor]
        item_score = out[item_tensor]
        score = torch.sum(user_score * item_score, dim=1).view(-1, 2)
        loss = -torch.mean(torch.log(torch.sigmoid(torch.matmul(score, self.weight))))
        
        reg_embedding_loss = (self.id_embedding[user_tensor]**2 + self.id_embedding[item_tensor]**2).mean()
        if self.v_feat is not None:
            reg_embedding_loss += (self.v_gcn.preference**2).mean()
        reg_loss = self.reg_weight * reg_embedding_loss
        
        total_loss = loss + reg_loss

        # Add propensity score loss if denoising is enabled
        if self.use_denoise and rf_outputs is not None and "ps_loss" in rf_outputs:
            total_loss = total_loss + self.ps_loss_weight * rf_outputs["ps_loss"]

        return total_loss

    def full_sort_predict(self, interaction):
        with torch.no_grad():
            if self.use_rf:
                # Forward pass with RF
                representation = None
                if self.v_feat is not None:
                    representation = self.v_gcn(self.v_feat, self.id_embedding)
                if self.t_feat is not None:
                    if representation is None:
                        representation = self.t_gcn(self.t_feat, self.id_embedding)
                    else:
                        representation += self.t_gcn(self.t_feat, self.id_embedding)
                representation /= self.num_modal

                # Prepare conditions
                conditions = []
                if self.v_feat is not None:
                    v_representation = self.v_gcn(self.v_feat, self.id_embedding)
                    conditions.append(v_representation)
                if self.t_feat is not None:
                    t_representation = self.t_gcn(self.t_feat, self.id_embedding)
                    conditions.append(t_representation)

                # Generate RF embeddings
                rf_embeds = self.rf_generator.generate(conditions)
                representation = self.rf_generator.mix_embeddings(
                    representation,
                    rf_embeds,
                    training=False,
                    epoch=self.rf_generator.current_epoch,
                )
                self.result = representation

        user_tensor = self.result[:self.n_users]
        item_tensor = self.result[self.n_users:]

        temp_user_tensor = user_tensor[interaction[0], :]
        score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())
        return score_matrix

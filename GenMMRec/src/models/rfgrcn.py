# coding: utf-8
# @email: enoche.chow@gmail.com
r"""
RFGRCN: RF-Enhanced GRCN
Integrates Rectified Flow module to enhance graph-refined convolutional embeddings
With optional causal denoising using Inverse Propensity Weighting (IPW)
"""

import torch
import torch.nn.functional as F

from models.grcn import GRCN
from models.rf_modules import RFEmbeddingGenerator, CausalDenoiser


class RFGRCN(GRCN):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.use_rf = config["use_rf"] if "use_rf" in config else True
        self.embedding_dim = config['embedding_size']
        self.latent_embedding = config['latent_embedding']

        if self.use_rf:
            # Calculate total embedding dimension (id + content)
            num_modal = 0
            if self.v_feat is not None:
                num_modal += 1
            if self.t_feat is not None:
                num_modal += 1
            
            # GRCN uses concat fusion: id_embedding + content_embeddings
            total_dim = self.embedding_dim + self.latent_embedding * num_modal

            # Initialize RF generator
            self.rf_generator = RFEmbeddingGenerator(
                embedding_dim=total_dim,
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
            num_modal = 0
            if self.v_feat is not None:
                num_modal += 1
            if self.t_feat is not None:
                num_modal += 1
            total_dim = self.embedding_dim + self.latent_embedding * num_modal
            
            self.causal_denoiser = CausalDenoiser(
                embedding_dim=total_dim,
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
        from torch_geometric.utils import dropout_adj
        
        weight = None
        content_rep = None
        num_modal = 0
        edge_index, _ = dropout_adj(self.edge_index, p=self.dropout)

        if self.v_feat is not None:
            num_modal += 1
            v_rep, weight_v = self.v_gcn(edge_index)
            weight = weight_v
            content_rep = v_rep

        if self.t_feat is not None:
            num_modal += 1
            t_rep, weight_t = self.t_gcn(edge_index)
            if weight is None:
                weight = weight_t   
                content_rep = t_rep
            else:
                content_rep = torch.cat((content_rep, t_rep), dim=1)
                if self.weight_mode == 'mean':  
                    weight = weight + weight_t
                else:
                    weight = torch.cat((weight, weight_t), dim=1)   

        if self.weight_mode == 'mean':
            weight = weight / num_modal
        elif self.weight_mode == 'max':
            weight, _ = torch.max(weight, dim=1)
            weight = weight.view(-1, 1)
        elif self.weight_mode == 'confid':
            confidence = torch.cat((self.model_specific_conf[edge_index[0]], self.model_specific_conf[edge_index[1]]), dim=0)
            weight = weight * confidence
            weight, _ = torch.max(weight, dim=1)
            weight = weight.view(-1, 1)

        if self.pruning:
            weight = torch.relu(weight)

        id_rep = self.id_gcn(edge_index, weight)

        if self.fusion_mode == 'concat':
            representation = torch.cat((id_rep, content_rep), dim=1)
        elif self.fusion_mode == 'id':
            representation = id_rep
        elif self.fusion_mode == 'mean':
            representation = (id_rep + content_rep) / 2

        # ===== RF Enhancement =====
        rf_outputs = None

        if self.use_rf and self.training:
            print(f"[RFGRCN] Forward in TRAINING mode")
            ps_loss = 0.0
            if self.use_denoise:
                # Use id_embedding for denoising (before graph convolution)
                ego_emb_for_denoise = self.id_gcn.id_embedding
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
                conditions.append(v_rep.detach())
            
            # Text features
            if self.t_feat is not None:
                conditions.append(t_rep.detach())

            # Compute user prior (routing-based preference deviation)
            if len(conditions) > 0:
                # Combine all modality representations
                combined_modal = torch.cat(conditions, dim=1) if len(conditions) > 1 else conditions[0]
                
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
            print(f"[RFGRCN] Forward in INFERENCE mode")
            # Inference mode
            with torch.no_grad():
                # Prepare multimodal conditions (need to recompute v_rep and t_rep)
                conditions = []
                if self.v_feat is not None:
                    conditions.append(v_rep)
                if self.t_feat is not None:
                    conditions.append(t_rep)
                
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
        
        reg_embedding_loss = (self.id_gcn.id_embedding[user_tensor]**2 + self.id_gcn.id_embedding[item_tensor]**2).mean()
        if self.v_feat is not None:
            reg_embedding_loss += (self.v_gcn.preference**2).mean()
        
        reg_content_loss = torch.zeros(1).to(self.device)
        if self.v_feat is not None:
            reg_content_loss = reg_content_loss + (self.v_gcn.preference[user_tensor]**2).mean()
        if self.t_feat is not None:            
            reg_content_loss = reg_content_loss + (self.t_gcn.preference[user_tensor]**2).mean()

        reg_loss = reg_embedding_loss + reg_content_loss
        reg_loss = self.reg_weight * reg_loss
        
        total_loss = loss + reg_loss

        # Add propensity score loss if denoising is enabled
        if self.use_denoise and rf_outputs is not None and "ps_loss" in rf_outputs:
            total_loss = total_loss + self.ps_loss_weight * rf_outputs["ps_loss"]

        return total_loss

    def full_sort_predict(self, interaction):
        with torch.no_grad():
            # Forward pass is already called during training, result is cached
            if not hasattr(self, 'result') or self.result is None:
                self.forward()

        user_tensor = self.result[:self.n_users]
        item_tensor = self.result[self.n_users:]

        temp_user_tensor = user_tensor[interaction[0], :]
        score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())
        return score_matrix

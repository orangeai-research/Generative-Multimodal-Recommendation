# coding: utf-8
# @email: enoche.chow@gmail.com
r"""
RFDualGNN: RF-Enhanced DualGNN
Integrates Rectified Flow module to enhance dual graph neural network embeddings
With optional causal denoising using Inverse Propensity Weighting (IPW)
"""

import torch
import torch.nn.functional as F

from models.dualgnn import DualGNN
from models.rf_modules import RFEmbeddingGenerator, CausalDenoiser


class RFDualGNN(DualGNN):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.use_rf = config["use_rf"] if "use_rf" in config else True
        self.embedding_dim = config['embedding_size']

        if self.use_rf:
            # Initialize RF generator
            self.rf_generator = RFEmbeddingGenerator(
                embedding_dim=self.dim_latent,
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
                embedding_dim=self.dim_latent,
                n_users=self.n_users,
                n_items=self.n_items,
                n_layers=config["denoise_layers"] if "denoise_layers" in config else 2,
                clean_rating_threshold=config["clean_rating_threshold"] if "clean_rating_threshold" in config else 5.0,
                device=self.device,
            )
            self.causal_denoiser.load_treatment_labels(dataset)

    def pre_epoch_processing(self):
        """Called by trainer at the beginning of each epoch."""
        super().pre_epoch_processing()  # Call parent's pre_epoch_processing
        
        if self.use_rf:
            self._training_epoch += 1
            self.rf_generator.set_epoch(self._training_epoch)
            self._rf_logged_this_epoch = False

    def forward(self, interaction):
        user_nodes, pos_item_nodes, neg_item_nodes = interaction[0], interaction[1], interaction[2]
        pos_item_nodes_shifted = pos_item_nodes + self.n_users
        neg_item_nodes_shifted = neg_item_nodes + self.n_users
        
        representation = None
        if self.v_feat is not None:
            self.v_rep, self.v_preference = self.v_gcn(self.edge_index_dropv, self.edge_index, self.v_feat)
            representation = self.v_rep
        if self.t_feat is not None:
            self.t_rep, self.t_preference = self.t_gcn(self.edge_index_dropt, self.edge_index, self.t_feat)
            if representation is None:
                representation = self.t_rep
            else:
                representation += self.t_rep

        # Weighted sum construction
        if self.construction == 'weighted_sum':
            if self.v_rep is not None:
                self.v_rep = torch.unsqueeze(self.v_rep, 2)
                user_rep = self.v_rep[:self.num_user]
            if self.t_rep is not None:
                self.t_rep = torch.unsqueeze(self.t_rep, 2)
                user_rep = self.t_rep[:self.num_user]
            if self.v_rep is not None and self.t_rep is not None:
                user_rep = torch.matmul(torch.cat((self.v_rep[:self.num_user], self.t_rep[:self.num_user]), dim=2),
                                        self.weight_u)
            user_rep = torch.squeeze(user_rep)

        item_rep = representation[self.num_user:]
        
        # User graph aggregation
        h_u1 = self.user_graph(user_rep, self.epoch_user_graph, self.user_weight_matrix)
        user_rep_final = user_rep + h_u1

        # ===== RF Enhancement =====
        rf_outputs = None

        if self.use_rf and self.training:
            print(f"[RFDualGNN] Forward in TRAINING mode")
            # Combine user and item representations
            all_rep = torch.cat((user_rep_final, item_rep), dim=0)
            
            ps_loss = 0.0
            if self.use_denoise:
                # Use preference embeddings for denoising
                if self.v_preference is not None:
                    v_pref_full = torch.cat([self.v_preference, torch.zeros(self.num_item, self.v_preference.shape[1]).to(self.device)], dim=0)
                    denoised_emb, ps_loss = self.causal_denoiser(v_pref_full)
                elif self.t_preference is not None:
                    t_pref_full = torch.cat([self.t_preference, torch.zeros(self.num_item, self.t_preference.shape[1]).to(self.device)], dim=0)
                    denoised_emb, ps_loss = self.causal_denoiser(t_pref_full)
                else:
                    denoised_emb = None
                    
                if denoised_emb is not None:
                    rf_target = denoised_emb.detach()
                else:
                    rf_target = all_rep.detach()
            else:
                rf_target = all_rep.detach()

            # Prepare multimodal conditions
            conditions = []
            
            # Visual features (GCN output)
            if self.v_feat is not None and self.v_rep is not None:
                v_rep_squeezed = torch.squeeze(self.v_rep) if self.v_rep.dim() > 2 else self.v_rep
                conditions.append(v_rep_squeezed.detach())
            
            # Text features (GCN output)
            if self.t_feat is not None and self.t_rep is not None:
                t_rep_squeezed = torch.squeeze(self.t_rep) if self.t_rep.dim() > 2 else self.t_rep
                conditions.append(t_rep_squeezed.detach())

            # Compute user prior (dual graph deviation)
            if len(conditions) > 0:
                # Combine all modality representations
                combined_modal = sum(conditions) / len(conditions)
                
                # Split user and item embeddings
                user_modal = combined_modal[:self.n_users]
                item_modal = combined_modal[self.n_users:]
                
                # User prior: deviation from user graph neighbors
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
            all_rep = self.rf_generator.mix_embeddings(
                all_rep,
                rf_embeds.detach(),
                training=True,
                epoch=self.rf_generator.current_epoch,
            )

            user_rep_final, item_rep = torch.split(all_rep, [self.n_users, self.n_items], dim=0)

            rf_outputs = {"ps_loss": ps_loss}
        elif self.use_rf and not self.training:
            print(f"[RFDualGNN] Forward in INFERENCE mode")
            # Inference mode
            with torch.no_grad():
                # Combine user and item representations
                all_rep = torch.cat((user_rep_final, item_rep), dim=0)
                
                # Prepare multimodal conditions
                conditions = []
                if self.v_feat is not None and self.v_rep is not None:
                    v_rep_squeezed = torch.squeeze(self.v_rep) if self.v_rep.dim() > 2 else self.v_rep
                    conditions.append(v_rep_squeezed)
                if self.t_feat is not None and self.t_rep is not None:
                    t_rep_squeezed = torch.squeeze(self.t_rep) if self.t_rep.dim() > 2 else self.t_rep
                    conditions.append(t_rep_squeezed)
                
                rf_embeds = self.rf_generator.generate(conditions)
                all_rep = self.rf_generator.mix_embeddings(
                    all_rep,
                    rf_embeds,
                    training=False,
                    epoch=self.rf_generator.current_epoch,
                )
                
                user_rep_final, item_rep = torch.split(all_rep, [self.n_users, self.n_items], dim=0)

        self.result_embed = torch.cat((user_rep_final, item_rep), dim=0)
        user_tensor = self.result_embed[user_nodes]
        pos_item_tensor = self.result_embed[pos_item_nodes_shifted]
        neg_item_tensor = self.result_embed[neg_item_nodes_shifted]
        pos_scores = torch.sum(user_tensor * pos_item_tensor, dim=1)
        neg_scores = torch.sum(user_tensor * neg_item_tensor, dim=1)
        
        if self.use_rf and self.training:
            return pos_scores, neg_scores, rf_outputs
        return pos_scores, neg_scores

    def calculate_loss(self, interaction):
        # Store batch indices for RF contrastive loss
        self._current_batch_users = interaction[0]
        self._current_batch_items = interaction[1]

        user = interaction[0]
        
        if self.use_rf:
            pos_scores, neg_scores, rf_outputs = self.forward(interaction)
        else:
            pos_scores, neg_scores = self.forward(interaction)
            rf_outputs = None
            
        loss_value = -torch.mean(torch.log2(torch.sigmoid(pos_scores - neg_scores)))
        
        reg_embedding_loss_v = (self.v_preference[user] ** 2).mean() if self.v_preference is not None else 0.0
        reg_embedding_loss_t = (self.t_preference[user] ** 2).mean() if self.t_preference is not None else 0.0

        reg_loss = self.reg_weight * (reg_embedding_loss_v + reg_embedding_loss_t)
        
        if self.construction == 'weighted_sum':
            reg_loss += self.reg_weight * (self.weight_u ** 2).mean()
            reg_loss += self.reg_weight * (self.weight_i ** 2).mean()
        elif self.construction == 'cat_mlp':
            reg_loss += self.reg_weight * (self.MLP_user.weight ** 2).mean()
            
        total_loss = loss_value + reg_loss

        # Add propensity score loss if denoising is enabled
        if self.use_denoise and rf_outputs is not None and "ps_loss" in rf_outputs:
            total_loss = total_loss + self.ps_loss_weight * rf_outputs["ps_loss"]

        return total_loss

    def full_sort_predict(self, interaction):
        # Result embed is already computed in forward pass
        user_tensor = self.result_embed[:self.n_users]
        item_tensor = self.result_embed[self.n_users:]

        temp_user_tensor = user_tensor[interaction[0], :]
        score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())
        return score_matrix

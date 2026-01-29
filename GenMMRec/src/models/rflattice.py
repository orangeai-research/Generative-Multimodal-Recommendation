# coding: utf-8
# @email: enoche.chow@gmail.com
r"""
RFLATTICE: RF-Enhanced LATTICE
Integrates Rectified Flow module to enhance latent structure mining for multimedia recommendation
With optional causal denoising using Inverse Propensity Weighting (IPW)
"""

import torch
import torch.nn.functional as F

from models.lattice import LATTICE
from models.rf_modules import RFEmbeddingGenerator, CausalDenoiser
from utils.utils import build_sim, compute_normalized_laplacian, build_knn_neighbourhood


class RFLATTICE(LATTICE):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.use_rf = config["use_rf"] if "use_rf" in config else True

        if self.use_rf:
            # Initialize RF generator
            self.rf_generator = RFEmbeddingGenerator(
                embedding_dim=self.embedding_dim,
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
                embedding_dim=self.embedding_dim,
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

    def forward(self, adj, build_item_graph=False):
        # Get multimodal features
        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
        if self.t_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight)
            
        if build_item_graph:
            weight = self.softmax(self.modal_weight)

            if self.v_feat is not None:
                self.image_adj = build_sim(image_feats)
                self.image_adj = build_knn_neighbourhood(self.image_adj, topk=self.knn_k)
                learned_adj = self.image_adj
                original_adj = self.image_original_adj
            if self.t_feat is not None:
                self.text_adj = build_sim(text_feats)
                self.text_adj = build_knn_neighbourhood(self.text_adj, topk=self.knn_k)
                learned_adj = self.text_adj
                original_adj = self.text_original_adj
            if self.v_feat is not None and self.t_feat is not None:
                learned_adj = weight[0] * self.image_adj + weight[1] * self.text_adj
                original_adj = weight[0] * self.image_original_adj + weight[1] * self.text_original_adj

            from utils.utils import compute_normalized_laplacian
            learned_adj = compute_normalized_laplacian(learned_adj)
            if self.item_adj is not None:
                del self.item_adj
            self.item_adj = (1 - self.lambda_coeff) * learned_adj + self.lambda_coeff * original_adj
        else:
            self.item_adj = self.item_adj.detach()

        h = self.item_id_embedding.weight
        for i in range(self.n_layers):
            h = torch.mm(self.item_adj, h)

        if self.cf_model == 'ngcf':
            ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
            all_embeddings = [ego_embeddings]
            for i in range(self.n_ui_layers):
                side_embeddings = torch.sparse.mm(adj, ego_embeddings)
                sum_embeddings = F.leaky_relu(self.GC_Linear_list[i](side_embeddings))
                bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
                bi_embeddings = F.leaky_relu(self.Bi_Linear_list[i](bi_embeddings))
                ego_embeddings = sum_embeddings + bi_embeddings
                ego_embeddings = self.dropout_list[i](ego_embeddings)

                norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
                all_embeddings += [norm_embeddings]

            all_embeddings = torch.stack(all_embeddings, dim=1)
            all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
            u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
            i_g_embeddings_ori = i_g_embeddings + F.normalize(h, p=2, dim=1)
            
        elif self.cf_model == 'lightgcn':
            ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
            all_embeddings = [ego_embeddings]
            for i in range(self.n_ui_layers):
                side_embeddings = torch.sparse.mm(adj, ego_embeddings)
                ego_embeddings = side_embeddings
                all_embeddings += [ego_embeddings]
            all_embeddings = torch.stack(all_embeddings, dim=1)
            all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
            u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
            i_g_embeddings_ori = i_g_embeddings + F.normalize(h, p=2, dim=1)
            
        elif self.cf_model == 'mf':
            u_g_embeddings = self.user_embedding.weight
            i_g_embeddings_ori = self.item_id_embedding.weight + F.normalize(h, p=2, dim=1)

        # ===== RF Enhancement =====
        rf_outputs = None

        if self.use_rf and self.training:
            print(f"[RFLATTICE] Forward in TRAINING mode")
            # Combine user and item embeddings
            all_embeddings_ori = torch.cat([u_g_embeddings, i_g_embeddings_ori], dim=0)
            
            ps_loss = 0.0
            if self.use_denoise:
                ego_emb_for_denoise = torch.cat([self.user_embedding.weight, self.item_id_embedding.weight], dim=0)
                denoised_emb, ps_loss = self.causal_denoiser(ego_emb_for_denoise)
                if denoised_emb is not None:
                    rf_target = denoised_emb.detach()
                else:
                    rf_target = all_embeddings_ori.detach()
            else:
                rf_target = all_embeddings_ori.detach()

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

            # Compute user prior (latent structure deviation)
            if len(conditions) > 0:
                # Combine all modality features
                combined_modal = sum(conditions) / len(conditions)
                
                # Split user and item features
                user_modal = combined_modal[:self.n_users]
                item_modal = combined_modal[self.n_users:]
                
                # User prior: deviation from average
                avg_user_modal = user_modal.mean(dim=0, keepdim=True)
                user_prior = user_modal - avg_user_modal
                
                # Item prior: deviation from average
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
            all_embeddings = self.rf_generator.mix_embeddings(
                all_embeddings_ori,
                rf_embeds.detach(),
                training=True,
                epoch=self.rf_generator.current_epoch,
            )

            u_g_embeddings, i_g_embeddings_ori = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)

            rf_outputs = {"ps_loss": ps_loss}
        elif self.use_rf and not self.training:
            print(f"[RFLATTICE] Forward in INFERENCE mode")
            # Inference mode
            with torch.no_grad():
                # Combine user and item embeddings
                all_embeddings_ori = torch.cat([u_g_embeddings, i_g_embeddings_ori], dim=0)
                
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
                all_embeddings = self.rf_generator.mix_embeddings(
                    all_embeddings_ori,
                    rf_embeds,
                    training=False,
                    epoch=self.rf_generator.current_epoch,
                )
                
                u_g_embeddings, i_g_embeddings_ori = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)

        if self.use_rf and self.training:
            return u_g_embeddings, i_g_embeddings_ori, rf_outputs
        return u_g_embeddings, i_g_embeddings_ori

    def calculate_loss(self, interaction):
        # Store batch indices for RF contrastive loss
        self._current_batch_users = interaction[0]
        self._current_batch_items = interaction[1]

        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        if self.use_rf:
            ua_embeddings, ia_embeddings, rf_outputs = self.forward(self.norm_adj, build_item_graph=self.build_item_graph)
        else:
            ua_embeddings, ia_embeddings = self.forward(self.norm_adj, build_item_graph=self.build_item_graph)
            rf_outputs = None
            
        self.build_item_graph = False

        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        batch_mf_loss, batch_emb_loss, batch_reg_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings,
                                                                      neg_i_g_embeddings)
        total_loss = batch_mf_loss + batch_emb_loss + batch_reg_loss

        # Add propensity score loss if denoising is enabled
        if self.use_denoise and rf_outputs is not None and "ps_loss" in rf_outputs:
            total_loss = total_loss + self.ps_loss_weight * rf_outputs["ps_loss"]

        return total_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]

        with torch.no_grad():
            if self.use_rf:
                ua_embeddings, ia_embeddings = self.forward(self.norm_adj, build_item_graph=True)
            else:
                ua_embeddings, ia_embeddings = self.forward(self.norm_adj, build_item_graph=True)

        u_embeddings = ua_embeddings[user]
        scores = torch.matmul(u_embeddings, ia_embeddings.transpose(0, 1))
        return scores

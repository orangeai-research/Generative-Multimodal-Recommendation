# coding: utf-8
# @email: enoche.chow@gmail.com
r"""
RFLightGCN: RF-Enhanced LightGCN
Integrates Rectified Flow module to enhance graph collaborative filtering embeddings
With optional causal denoising using Inverse Propensity Weighting (IPW)
"""

import torch
import torch.nn.functional as F

from models.lightgcn import LightGCN
from models.rf_modules import RFEmbeddingGenerator, CausalDenoiser


class RFLightGCN(LightGCN):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.use_rf = config["use_rf"] if "use_rf" in config else True

        if self.use_rf:
            # Initialize RF generator
            self.rf_generator = RFEmbeddingGenerator(
                embedding_dim=self.latent_dim,
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
                embedding_dim=self.latent_dim,
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
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        # ===== RF Enhancement =====
        rf_outputs = None

        if self.use_rf and self.training:
            print(f"[RFLightGCN] Forward in TRAINING mode")
            ps_loss = 0.0
            if self.use_denoise:
                ego_emb_for_denoise = self.get_ego_embeddings()
                denoised_emb, ps_loss = self.causal_denoiser(ego_emb_for_denoise)
                if denoised_emb is not None:
                    rf_target = denoised_emb.detach()
                else:
                    rf_target = lightgcn_all_embeddings.detach()
            else:
                rf_target = lightgcn_all_embeddings.detach()

            loss_dict = self.rf_generator.compute_loss_and_step(
                target_embeds=rf_target,
                conditions=[],
                user_prior=None,
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

            rf_embeds = self.rf_generator.generate([])
            lightgcn_all_embeddings = self.rf_generator.mix_embeddings(
                lightgcn_all_embeddings,
                rf_embeds.detach(),
                training=True,
                epoch=self.rf_generator.current_epoch,
            )

            rf_outputs = {"ps_loss": ps_loss}
        elif self.use_rf and not self.training:
            print(f"[RFLightGCN] Forward in INFERENCE mode")
            # Inference mode
            with torch.no_grad():
                rf_embeds = self.rf_generator.generate([])
                lightgcn_all_embeddings = self.rf_generator.mix_embeddings(
                    lightgcn_all_embeddings,
                    rf_embeds,
                    training=False,
                    epoch=self.rf_generator.current_epoch,
                )

        user_all_embeddings = lightgcn_all_embeddings[:self.n_users, :]
        item_all_embeddings = lightgcn_all_embeddings[self.n_users:, :]

        if self.use_rf and self.training:
            return user_all_embeddings, item_all_embeddings, rf_outputs
        return user_all_embeddings, item_all_embeddings

    def calculate_loss(self, interaction):
        self._current_batch_users = interaction[0]
        self._current_batch_items = interaction[1]

        user = interaction[0]
        pos_item = interaction[1]
        neg_item = interaction[2]

        if self.use_rf:
            user_all_embeddings, item_all_embeddings, rf_outputs = self.forward()
        else:
            user_all_embeddings, item_all_embeddings = self.forward()
            rf_outputs = None

        u_embeddings = user_all_embeddings[user, :]
        posi_embeddings = item_all_embeddings[pos_item, :]
        negi_embeddings = item_all_embeddings[neg_item, :]

        pos_scores = torch.mul(u_embeddings, posi_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, negi_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        u_ego_embeddings = self.embedding_dict['user_emb'][user, :]
        posi_ego_embeddings = self.embedding_dict['item_emb'][pos_item, :]
        negi_ego_embeddings = self.embedding_dict['item_emb'][neg_item, :]

        reg_loss = self.reg_loss(u_ego_embeddings, posi_ego_embeddings, negi_ego_embeddings)
        total_loss = mf_loss + self.reg_weight * reg_loss

        if self.use_denoise and rf_outputs is not None and "ps_loss" in rf_outputs:
            total_loss = total_loss + self.ps_loss_weight * rf_outputs["ps_loss"]

        return total_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        
        with torch.no_grad():
            if self.use_rf:
                all_embeddings = self.get_ego_embeddings()
                embeddings_list = [all_embeddings]
                for layer_idx in range(self.n_layers):
                    all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
                    embeddings_list.append(all_embeddings)
                lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
                lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

                rf_embeds = self.rf_generator.generate([])
                lightgcn_all_embeddings = self.rf_generator.mix_embeddings(
                    lightgcn_all_embeddings,
                    rf_embeds,
                    training=False,
                    epoch=self.rf_generator.current_epoch,
                )

                restore_user_e = lightgcn_all_embeddings[:self.n_users, :]
                restore_item_e = lightgcn_all_embeddings[self.n_users:, :]
            else:
                restore_user_e, restore_item_e = self.forward()

        u_embeddings = restore_user_e[user, :]
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores

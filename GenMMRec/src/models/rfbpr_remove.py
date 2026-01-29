# coding: utf-8
# @email: enoche.chow@gmail.com
r"""
RFBPR: RF-Enhanced BPR
Integrates Rectified Flow module to enhance collaborative filtering embeddings
With optional causal denoising using Inverse Propensity Weighting (IPW)
"""

import torch
import torch.nn.functional as F

from models.bpr import BPR
from models.rf_modules import RFEmbeddingGenerator, CausalDenoiser


class RFBPR(BPR):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.use_rf = config["use_rf"] if "use_rf" in config else True

        if self.use_rf:
            # Initialize RF generator with consistent parameters
            self.rf_generator = RFEmbeddingGenerator(
                embedding_dim=self.embedding_size,
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
                # 2-RF parameters
                use_2rf=config["use_2rf"] if "use_2rf" in config else False,
                rf_2rf_transition_epoch=config["rf_2rf_transition_epoch"] if "rf_2rf_transition_epoch" in config else None,
                # Memory optimization
                use_gradient_checkpointing=config["use_gradient_checkpointing"] if "use_gradient_checkpointing" in config else True,
            )
            self._rf_logged_this_epoch = False

            # Store batch indices for RF contrastive loss
            self._current_batch_users = None
            self._current_batch_items = None

            # Track training epoch
            self._training_epoch = -1

        # ===== Denoising Module =====
        self.use_denoise = config["use_denoise"] if "use_denoise" in config else False

        if self.use_denoise:
            self.ps_loss_weight = config["ps_loss_weight"] if "ps_loss_weight" in config else 0.1

            # Initialize CausalDenoiser
            self.causal_denoiser = CausalDenoiser(
                embedding_dim=self.embedding_size,
                n_users=self.n_users,
                n_items=self.n_items,
                n_layers=config["denoise_layers"] if "denoise_layers" in config else 2,
                clean_rating_threshold=config["clean_rating_threshold"] if "clean_rating_threshold" in config else 5.0,
                device=self.device,
            )
            # Load treatment labels from dataset
            self.causal_denoiser.load_treatment_labels(dataset)

    def pre_epoch_processing(self):
        """Called by trainer at the beginning of each epoch."""
        if self.use_rf:
            self._training_epoch += 1
            self.rf_generator.set_epoch(self._training_epoch)
            self._rf_logged_this_epoch = False

    def forward(self, dropout=0.0):
        user_e = F.dropout(self.user_embedding.weight, dropout)
        item_e = F.dropout(self.item_embedding.weight, dropout)

        # Combine user and item embeddings
        all_embeddings_ori = torch.cat([user_e, item_e], dim=0)

        # ===== RF Enhancement =====
        rf_outputs = None

        if self.use_rf and self.training:
            print(f"[RFBPR] Forward in TRAINING mode")
            # ===== Denoising: compute denoised embeddings as RF target =====
            ps_loss = 0.0
            if self.use_denoise:
                # Get initial ego embeddings for denoising
                ego_emb_for_denoise = torch.cat((self.user_embedding.weight, self.item_embedding.weight), dim=0)
                denoised_emb, ps_loss = self.causal_denoiser(ego_emb_for_denoise)
                if denoised_emb is not None:
                    rf_target = denoised_emb.detach()
                else:
                    rf_target = all_embeddings_ori.detach()
            else:
                rf_target = all_embeddings_ori.detach()

            # RF training (no multimodal conditions for BPR, use zero conditions)
            loss_dict = self.rf_generator.compute_loss_and_step(
                target_embeds=rf_target,
                conditions=[],  # No multimodal features for BPR
                user_prior=None,  # No prior for simple CF
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
            rf_embeds = self.rf_generator.generate([])

            # Mix original and RF generated embeddings
            all_embeddings = self.rf_generator.mix_embeddings(
                all_embeddings_ori,
                rf_embeds.detach(),
                training=True,
                epoch=self.rf_generator.current_epoch,
            )

            rf_outputs = {"ps_loss": ps_loss}
        elif self.use_rf and not self.training:
            print(f"[RFBPR] Forward in INFERENCE mode")
            # Inference mode
            with torch.no_grad():
                rf_embeds = self.rf_generator.generate([])
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
                # Generate RF embeddings for inference
                all_embeddings_ori = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
                rf_embeds = self.rf_generator.generate([])
                all_embeddings = self.rf_generator.mix_embeddings(
                    all_embeddings_ori,
                    rf_embeds,
                    training=False,
                    epoch=self.rf_generator.current_epoch,
                )
                user_e, all_item_e = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
                user_e = user_e[user]
            else:
                user_e = self.user_embedding.weight[user]
                all_item_e = self.item_embedding.weight

        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score

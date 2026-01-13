# coding: utf-8
# @email: enoche.chow@gmail.com
r"""
RFBM3: RF-Enhanced BM3
Integrates Rectified Flow module to enhance collaborative filtering embeddings
"""

import torch
import torch.nn.functional as F

from models.bm3 import BM3
from models.rf_modules import RFEmbeddingGenerator


class RFBM3(BM3):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.use_rf = config["use_rf"] if "use_rf" in config else True

        if self.use_rf:
            # Initialize RF generator with consistent parameters
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
            )
            self._rf_logged_this_epoch = False

    def set_epoch(self, epoch):
        """Set current epoch for RF generator."""
        if self.use_rf:
            self.rf_generator.set_epoch(epoch)
            self._rf_logged_this_epoch = False

    def forward(self):
        h = self.item_id_embedding.weight

        # Original BM3 GCN
        ego_embeddings = torch.cat((self.user_embedding.weight,
                                    self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]

        for i in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)

        u_g_embeddings, i_g_embeddings_ori = torch.split(
            all_embeddings, [self.n_users, self.n_items], dim=0
        )

        # ===== RF Enhancement =====
        i_g_embeddings = i_g_embeddings_ori

        if self.use_rf:
            # Prepare multimodal conditions
            t_feat_online = self.text_trs(self.text_embedding.weight) if self.t_feat is not None else None
            v_feat_online = self.image_trs(self.image_embedding.weight) if self.v_feat is not None else None

            conditions = []
            if v_feat_online is not None:
                conditions.append(v_feat_online)
            if t_feat_online is not None:
                conditions.append(t_feat_online)

            if len(conditions) > 0 and self.training:
                # RF training
                loss_dict = self.rf_generator.compute_loss_and_step(
                    target_embeds=i_g_embeddings_ori.detach(),
                    conditions=[c.detach() for c in conditions],
                )

                if not self._rf_logged_this_epoch:
                    print(f"  [RF Train] epoch={self.rf_generator.current_epoch}, "
                          f"rf_loss={loss_dict['rf_loss']:.6f}, "
                          f"cl_loss={loss_dict['cl_loss']:.6f}")
                    self._rf_logged_this_epoch = True

                # Generate and mix
                rf_embeds = self.rf_generator.generate(conditions)
                i_g_embeddings = self.rf_generator.mix_embeddings(
                    i_g_embeddings_ori, rf_embeds.detach(), training=True
                )

            elif len(conditions) > 0 and not self.training:
                # Inference mode
                with torch.no_grad():
                    rf_embeds = self.rf_generator.generate(conditions)
                    i_g_embeddings = self.rf_generator.mix_embeddings(
                        i_g_embeddings_ori, rf_embeds, training=False
                    )

        # Residual connection
        i_g_embeddings = i_g_embeddings + h

        return u_g_embeddings, i_g_embeddings

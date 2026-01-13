# coding: utf-8
# @email: georgeguo.gzq.cn@gmail.com
r"""
RFLGMRec: RF-Enhanced LGMRec
Integrates Rectified Flow module to enhance collaborative graph embeddings
"""

import torch
import torch.nn.functional as F

from models.lgmrec import LGMRec
from models.rf_modules import RFEmbeddingGenerator


class RFLGMRec(LGMRec):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.use_rf = config["use_rf"] if "use_rf" in config else True

        if self.use_rf:
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
        # hyperedge dependencies constructing
        if self.v_feat is not None:
            iv_hyper = torch.mm(self.image_embedding.weight, self.v_hyper)
            uv_hyper = torch.mm(self.adj, iv_hyper)
            iv_hyper = F.gumbel_softmax(iv_hyper, self.tau, dim=1, hard=False)
            uv_hyper = F.gumbel_softmax(uv_hyper, self.tau, dim=1, hard=False)
        if self.t_feat is not None:
            it_hyper = torch.mm(self.text_embedding.weight, self.t_hyper)
            ut_hyper = torch.mm(self.adj, it_hyper)
            it_hyper = F.gumbel_softmax(it_hyper, self.tau, dim=1, hard=False)
            ut_hyper = F.gumbel_softmax(ut_hyper, self.tau, dim=1, hard=False)

        # CGE: collaborative graph embedding
        cge_embs = self.cge()

        # Split user and item CGE
        u_cge, i_cge = torch.split(cge_embs, [self.n_users, self.n_items], dim=0)

        # ===== RF Enhancement for item CGE =====
        if self.use_rf:
            conditions = []

            # Get modality features
            if self.v_feat is not None:
                v_feats = self.mge('v')  # Already aggregated on user-item graph
                v_feats_item = v_feats[self.n_users:]  # Only take item part
                conditions.append(v_feats_item)

            if self.t_feat is not None:
                t_feats = self.mge('t')  # Already aggregated on user-item graph
                t_feats_item = t_feats[self.n_users:]  # Only take item part
                conditions.append(t_feats_item)

            if len(conditions) > 0 and self.training:
                # RF training
                loss_dict = self.rf_generator.compute_loss_and_step(
                    target_embeds=i_cge.detach(),
                    conditions=[c.detach() for c in conditions],
                )

                if not self._rf_logged_this_epoch:
                    print(f"  [RF Train] epoch={self.rf_generator.current_epoch}, "
                          f"rf_loss={loss_dict['rf_loss']:.6f}, "
                          f"cl_loss={loss_dict['cl_loss']:.6f}")
                    self._rf_logged_this_epoch = True

                # Generate and mix
                rf_embeds = self.rf_generator.generate(conditions)
                i_cge = self.rf_generator.mix_embeddings(
                    i_cge, rf_embeds.detach(), training=True
                )

            elif len(conditions) > 0 and not self.training:
                # Inference mode
                with torch.no_grad():
                    rf_embeds = self.rf_generator.generate(conditions)
                    i_cge = self.rf_generator.mix_embeddings(
                        i_cge, rf_embeds, training=False
                    )

        # Recombine CGE with RF-enhanced item embeddings
        cge_embs = torch.cat([u_cge, i_cge], dim=0)

        # Continue with original LGMRec logic
        if self.v_feat is not None and self.t_feat is not None:
            # MGE: modal graph embedding
            v_feats = self.mge('v')
            t_feats = self.mge('t')
            # local embeddings = collaborative-related embedding + modality-related embedding
            mge_embs = F.normalize(v_feats) + F.normalize(t_feats)
            lge_embs = cge_embs + mge_embs
            # GHE: global hypergraph embedding
            uv_hyper_embs, iv_hyper_embs = self.hgnnLayer(self.drop(iv_hyper), self.drop(uv_hyper), cge_embs[self.n_users:])
            ut_hyper_embs, it_hyper_embs = self.hgnnLayer(self.drop(it_hyper), self.drop(ut_hyper), cge_embs[self.n_users:])
            av_hyper_embs = torch.concat([uv_hyper_embs, iv_hyper_embs], dim=0)
            at_hyper_embs = torch.concat([ut_hyper_embs, it_hyper_embs], dim=0)
            ghe_embs = av_hyper_embs + at_hyper_embs
            # local embeddings + alpha * global embeddings
            all_embs = lge_embs + self.alpha * F.normalize(ghe_embs)
        else:
            all_embs = cge_embs

        u_embs, i_embs = torch.split(all_embs, [self.n_users, self.n_items], dim=0)

        return u_embs, i_embs, [uv_hyper_embs, iv_hyper_embs, ut_hyper_embs, it_hyper_embs]

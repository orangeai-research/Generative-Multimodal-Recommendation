# coding: utf-8
# @email: enoche.chow@gmail.com

r"""
################################
"""

import os
import itertools
import torch
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import scipy.sparse as sp
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt

from time import time
from logging import getLogger

from utils.utils import get_local_time, early_stopping, dict2str
from utils.topk_evaluator import TopKEvaluator

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class AbstractTrainer(object):
    r"""Trainer Class is used to manage the training and evaluation processes of recommender system models.
    AbstractTrainer is an abstract class in which the fit() and evaluate() method should be implemented according
    to different training and evaluation strategies.
    """

    def __init__(self, config, model):
        self.config = config
        self.model = model

    def fit(self, train_data):
        r"""Train the model based on the train data.

        """
        raise NotImplementedError('Method [next] should be implemented.')

    def evaluate(self, eval_data):
        r"""Evaluate the model based on the eval data.

        """

        raise NotImplementedError('Method [next] should be implemented.')


class Trainer(AbstractTrainer):
    r"""The basic Trainer for basic training and evaluation strategies in recommender systems. This class defines common
    functions for training and evaluation processes of most recommender system models, including fit(), evaluate(),
    and some other features helpful for model training and evaluation.

    Generally speaking, this class can serve most recommender system models, If the training process of the model is to
    simply optimize a single loss without involving any complex training strategies, such as adversarial learning,
    pre-training and so on.

    Initializing the Trainer needs two parameters: `config` and `model`. `config` records the parameters information
    for controlling training and evaluation, such as `learning_rate`, `epochs`, `eval_step` and so on.
    More information can be found in [placeholder]. `model` is the instantiated object of a Model Class.

    """

    def __init__(self, config, model, mg=False):
        super(Trainer, self).__init__(config, model)

        self.logger = getLogger()
        self.learner = config['learner']
        self.learning_rate = config['learning_rate']
        self.epochs = config['epochs']
        self.eval_step = min(config['eval_step'], self.epochs)
        self.stopping_step = config['stopping_step']
        self.clip_grad_norm = config['clip_grad_norm']
        self.valid_metric = config['valid_metric'].lower()
        self.valid_metric_bigger = config['valid_metric_bigger']
        self.test_batch_size = config['eval_batch_size']
        self.device = config['device']
        self.weight_decay = 0.0
        if config['weight_decay'] is not None:
            wd = config['weight_decay']
            self.weight_decay = eval(wd) if isinstance(wd, str) else wd

        self.req_training = config['req_training']

        # W&B logging
        self.use_wandb = (config['use_wandb'] if 'use_wandb' in config else True) and WANDB_AVAILABLE and wandb.run is not None

        self.start_epoch = 0
        self.cur_step = 0

        tmp_dd = {}
        for j, k in list(itertools.product(config['metrics'], config['topk'])):
            tmp_dd[f'{j.lower()}@{k}'] = 0.0
        self.best_valid_score = -1
        self.best_valid_result = tmp_dd
        self.best_test_upon_valid = tmp_dd
        self.train_loss_dict = dict()
        self.optimizer = self._build_optimizer()

        #fac = lambda epoch: 0.96 ** (epoch / 50)
        lr_scheduler = config['learning_rate_scheduler']        # check zero?
        fac = lambda epoch: lr_scheduler[0] ** (epoch / lr_scheduler[1])
        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=fac)
        self.lr_scheduler = scheduler

        self.eval_type = config['eval_type']
        self.evaluator = TopKEvaluator(config)

        self.item_tensor = None
        self.tot_item_num = None
        self.mg = mg
        self.alpha1 = config['alpha1']
        self.alpha2 = config['alpha2']
        self.beta = config['beta']

    def _build_optimizer(self):
        r"""Init the Optimizer

        Returns:
            torch.optim: the optimizer
        """
        if self.learner.lower() == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'adagrad':
            optimizer = optim.Adagrad(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'rmsprop':
            optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            self.logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def _train_epoch(self, train_data, epoch_idx, loss_func=None):
        r"""Train the model in an epoch

        Args:
            train_data (DataLoader): The train data.
            epoch_idx (int): The current epoch id.
            loss_func (function): The loss function of :attr:`model`. If it is ``None``, the loss function will be
                :attr:`self.model.calculate_loss`. Defaults to ``None``.

        Returns:
            float/tuple: The sum of loss returned by all batches in this epoch. If the loss in each batch contains
            multiple parts and the model return these multiple parts loss instead of the sum of loss, It will return a
            tuple which includes the sum of loss in each part.
        """
        if not self.req_training:
            return 0.0, []
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        loss_batches = []
        for batch_idx, interaction in enumerate(train_data):
            self.optimizer.zero_grad()
            second_inter = interaction.clone()
            losses = loss_func(interaction)
            
            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            if self._check_nan(loss):
                self.logger.info('Loss is nan at epoch: {}, batch index: {}. Exiting.'.format(epoch_idx, batch_idx))
                return loss, torch.tensor(0.0)
            
            if self.mg and batch_idx % self.beta == 0:
                first_loss = self.alpha1 * loss
                first_loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()
                
                losses = loss_func(second_inter)
                if isinstance(losses, tuple):
                    loss = sum(losses)
                else:
                    loss = losses
                    
                if self._check_nan(loss):
                    self.logger.info('Loss is nan at epoch: {}, batch index: {}. Exiting.'.format(epoch_idx, batch_idx))
                    return loss, torch.tensor(0.0)
                second_loss = -1 * self.alpha2 * loss
                second_loss.backward()
            else:
                loss.backward()
                
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            self.optimizer.step()
            loss_batches.append(loss.detach())
            # for test
            #if batch_idx == 0:
            #    break
        return total_loss, loss_batches

    def _valid_epoch(self, valid_data):
        r"""Valid the model with valid data

        Args:
            valid_data (DataLoader): the valid data

        Returns:
            float: valid score
            dict: valid result
        """
        valid_result = self.evaluate(valid_data)
        valid_score = valid_result[self.valid_metric] if self.valid_metric else valid_result['NDCG@20']
        return valid_score, valid_result

    def _check_nan(self, loss):
        if torch.isnan(loss):
            #raise ValueError('Training loss is nan')
            return True

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, losses):
        train_loss_output = 'epoch %d training [time: %.2fs, ' % (epoch_idx, e_time - s_time)
        if isinstance(losses, tuple):
            train_loss_output = ', '.join('train_loss%d: %.4f' % (idx + 1, loss) for idx, loss in enumerate(losses))
        else:
            train_loss_output += 'train loss: %.4f' % losses
        return train_loss_output + ']'

    def fit(self, train_data, valid_data=None, test_data=None, saved=False, verbose=True):
        r"""Train the model based on the train data and the valid data.

        Args:
            train_data (DataLoader): the train data
            valid_data (DataLoader, optional): the valid data, default: None.
                                               If it's None, the early_stopping is invalid.
            test_data (DataLoader, optional): None
            verbose (bool, optional): whether to write training and evaluation information to logger, default: True
            saved (bool, optional): whether to save the model parameters, default: True

        Returns:
             (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        for epoch_idx in range(self.start_epoch, self.epochs):
            # train
            training_start_time = time()
            self.model.pre_epoch_processing()
            train_loss, _ = self._train_epoch(train_data, epoch_idx)
            if torch.is_tensor(train_loss):
                # get nan loss
                break
            #for param_group in self.optimizer.param_groups:
            #    print('======lr: ', param_group['lr'])
            self.lr_scheduler.step()

            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            post_info = self.model.post_epoch_processing()
            if verbose:
                self.logger.info(train_loss_output)
                if post_info is not None:
                    self.logger.info(post_info)

            # Log training loss to wandb
            if self.use_wandb:
                wandb_log = {
                    'epoch': epoch_idx,
                    'train/time': training_end_time - training_start_time,
                    'train/lr': self.optimizer.param_groups[0]['lr']
                }
                if isinstance(train_loss, tuple):
                    for idx, loss in enumerate(train_loss):
                        wandb_log[f'train/loss_{idx+1}'] = loss
                    wandb_log['train/total_loss'] = sum(train_loss)
                else:
                    wandb_log['train/loss'] = train_loss
                wandb.log(wandb_log, step=epoch_idx)

            # eval: To ensure the test result is the best model under validation data, set self.eval_step == 1
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(valid_data)
                self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                    valid_score, self.best_valid_score, self.cur_step,
                    max_step=self.stopping_step, bigger=self.valid_metric_bigger)
                valid_end_time = time()
                valid_score_output = "epoch %d evaluating [time: %.2fs, valid_score: %f]" % \
                                     (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = 'valid result: \n' + dict2str(valid_result)
                # test
                _, test_result = self._valid_epoch(test_data)
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                    self.logger.info('test result: \n' + dict2str(test_result))

                # Log validation and test results to wandb
                if self.use_wandb:
                    wandb_eval_log = {
                        'epoch': epoch_idx,
                        'valid/score': valid_score,
                        'valid/time': valid_end_time - valid_start_time,
                        **{f'valid/{k}': v for k, v in valid_result.items()},
                        **{f'test/{k}': v for k, v in test_result.items()}
                    }
                    wandb.log(wandb_eval_log, step=epoch_idx)

                if update_flag:
                    update_output = '██ ' + self.config['model'] + '--Best validation results updated!!!'
                    if verbose:
                        self.logger.info(update_output)
                    self.best_valid_result = valid_result
                    self.best_test_upon_valid = test_result

                    # Log best results to wandb
                    if self.use_wandb:
                        wandb.run.summary.update({
                            'best_epoch': epoch_idx,
                            'best_valid_score': valid_score,
                            **{f'best_valid_{k}': v for k, v in valid_result.items()},
                            **{f'best_test_{k}': v for k, v in test_result.items()}
                        })

                if stop_flag:
                    stop_output = '+++++Finished training, best eval result in epoch %d' % \
                                  (epoch_idx - self.cur_step * self.eval_step)
                    if verbose:
                        self.logger.info(stop_output)
                    break
        return self.best_valid_score, self.best_valid_result, self.best_test_upon_valid


    @torch.no_grad()
    def evaluate(self, eval_data, is_test=False, idx=0):
        r"""Evaluate the model based on the eval data.
        Returns:
            dict: eval result, key is the eval metric and value in the corresponding metric value
        """
        self.model.eval()

        # batch full users
        batch_matrix_list = []
        for batch_idx, batched_data in enumerate(eval_data):
            # predict: interaction without item ids
            scores = self.model.full_sort_predict(batched_data)
            masked_items = batched_data[1]
            # mask out pos items
            scores[masked_items[0], masked_items[1]] = -1e10
            # rank and get top-k
            _, topk_index = torch.topk(scores, max(self.config['topk']), dim=-1)  # nusers x topk
            batch_matrix_list.append(topk_index)
        return self.evaluator.evaluate(batch_matrix_list, eval_data, is_test=is_test, idx=idx)

    def plot_train_loss(self, show=True, save_path=None):
        r"""Plot the train loss in each epoch

        Args:
            show (bool, optional): whether to show this figure, default: True
            save_path (str, optional): the data path to save the figure, default: None.
                                       If it's None, it will not be saved.
        """
        epochs = list(self.train_loss_dict.keys())
        epochs.sort()
        values = [float(self.train_loss_dict[epoch]) for epoch in epochs]
        plt.plot(epochs, values)
        plt.xticks(epochs)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        if show:
            plt.show()
        if save_path:
            plt.savefig(save_path)

class DiffMMTrainer(Trainer):
    def __init__(self, config, model, mg=False):
        super(DiffMMTrainer, self).__init__(config, model, mg)
        
        self.denoise_opt_image = optim.Adam(self.model.denoise_model_image.parameters(), lr=config['learning_rate'], weight_decay=0)
        self.denoise_opt_text = optim.Adam(self.model.denoise_model_text.parameters(), lr=config['learning_rate'], weight_decay=0)
        
        self.diffusion_loader = None
        self.item_num = model.n_items
        self.user_num = model.n_users

    def _build_diffusion_loader(self, train_data):
        if self.diffusion_loader is not None:
            return
            
        # Extract user interactions
        df = train_data.dataset.df
        uid_field = self.config['USER_ID_FIELD']
        iid_field = self.config['ITEM_ID_FIELD']
        
        user_interactions = df.groupby(uid_field)[iid_field].apply(list).to_dict()
        
        # Create dense vectors
        # Note: This might consume memory. If user_num * item_num is large.
        # DiffMM uses sparse conversion in DataHandler but DataLoader yields dense tensors?
        # Let's check DiffMM logic: 
        # "return torch.FloatTensor(itemTensor), torch.tensor(index).float()" in DiffusionData
        # So it returns dense tensors.
        
        # We can implement a custom Dataset
        class DiffusionDataset(torch.utils.data.Dataset):
            def __init__(self, user_interactions, user_num, item_num):
                self.user_interactions = user_interactions
                self.user_num = user_num
                self.item_num = item_num
                self.users = list(range(user_num))
                
            def __len__(self):
                return self.user_num
                
            def __getitem__(self, idx):
                user_id = self.users[idx]
                items = self.user_interactions.get(user_id, [])
                
                # Create dense vector
                vector = torch.zeros(self.item_num)
                if len(items) > 0:
                    vector[items] = 1.0
                
                return vector, torch.tensor(user_id).float()
                
        dataset = DiffusionDataset(user_interactions, self.user_num, self.item_num)
        self.diffusion_loader = DataLoader(dataset, batch_size=self.config['train_batch_size'], shuffle=True, num_workers=0) # num_workers=0 for safety

    def normalizeAdj(self, mat): 
        degree = np.array(mat.sum(axis=-1))
        dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
        dInvSqrt[np.isinf(dInvSqrt)] = 0.0
        dInvSqrtMat = sp.diags(dInvSqrt)
        return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

    def buildUIMatrix(self, u_list, i_list, edge_list):
        mat = coo_matrix((edge_list, (u_list, i_list)), shape=(self.user_num, self.item_num), dtype=np.float32)

        a = sp.csr_matrix((self.user_num, self.user_num))
        b = sp.csr_matrix((self.item_num, self.item_num))
        mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
        mat = (mat != 0) * 1.0
        mat = (mat + sp.eye(mat.shape[0])) * 1.0
        mat = self.normalizeAdj(mat)

        idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = torch.from_numpy(mat.data.astype(np.float32))
        shape = torch.Size(mat.shape)

        return torch.sparse.FloatTensor(idxs, vals, shape).to(self.device)

    def _train_epoch(self, train_data, epoch_idx, loss_func=None):
        # 0. Build diffusion loader if not exists
        self._build_diffusion_loader(train_data)
        
        # 1. Diffusion Training
        self.model.train()
        epDiLoss_image, epDiLoss_text = 0, 0
        steps = len(self.diffusion_loader)
        
        iEmbeds = self.model.getItemEmbeds().detach()
        # uEmbeds not used in diffusion training in DiffMM? 
        # DiffMM: iEmbeds = self.model.getItemEmbeds().detach()
        # usr_id_embeds = torch.mm(x_start, itmEmbeds)
        
        image_feats = self.model.getImageFeats().detach()
        text_feats = self.model.getTextFeats().detach()
        
        for i, batch in enumerate(self.diffusion_loader):
            batch_item, batch_index = batch
            batch_item, batch_index = batch_item.to(self.device), batch_index.to(self.device)
            
            self.denoise_opt_image.zero_grad()
            self.denoise_opt_text.zero_grad()
            
            diff_loss_image, gc_loss_image = self.model.diffusion_model.training_losses(
                self.model.denoise_model_image, batch_item, iEmbeds, batch_index, image_feats)
            
            diff_loss_text, gc_loss_text = self.model.diffusion_model.training_losses(
                self.model.denoise_model_text, batch_item, iEmbeds, batch_index, text_feats)
                
            loss_image = diff_loss_image.mean() + gc_loss_image.mean() * self.model.e_loss
            loss_text = diff_loss_text.mean() + gc_loss_text.mean() * self.model.e_loss
            
            epDiLoss_image += loss_image.item()
            epDiLoss_text += loss_text.item()
            
            loss = loss_image + loss_text
            loss.backward()
            
            self.denoise_opt_image.step()
            self.denoise_opt_text.step()
            
        # 2. Graph Construction
        # Re-generate matrices using trained diffusion model
        with torch.no_grad():
            u_list_image = []
            i_list_image = []
            edge_list_image = []

            u_list_text = []
            i_list_text = []
            edge_list_text = []
            
            for _, batch in enumerate(self.diffusion_loader):
                batch_item, batch_index = batch
                batch_item, batch_index = batch_item.to(self.device), batch_index.to(self.device)

                # image
                denoised_batch = self.model.diffusion_model.p_sample(self.model.denoise_model_image, batch_item, self.model.sampling_steps, self.model.sampling_noise)
                top_item, indices_ = torch.topk(denoised_batch, k=self.model.rebuild_k)

                for i in range(batch_index.shape[0]):
                    for j in range(indices_[i].shape[0]): 
                        u_list_image.append(int(batch_index[i].cpu().numpy()))
                        i_list_image.append(int(indices_[i][j].cpu().numpy()))
                        edge_list_image.append(1.0)

                # text
                denoised_batch = self.model.diffusion_model.p_sample(self.model.denoise_model_text, batch_item, self.model.sampling_steps, self.model.sampling_noise)
                top_item, indices_ = torch.topk(denoised_batch, k=self.model.rebuild_k)

                for i in range(batch_index.shape[0]):
                    for j in range(indices_[i].shape[0]): 
                        u_list_text.append(int(batch_index[i].cpu().numpy()))
                        i_list_text.append(int(indices_[i][j].cpu().numpy()))
                        edge_list_text.append(1.0)
            
            # image
            u_list_image = np.array(u_list_image)
            i_list_image = np.array(i_list_image)
            edge_list_image = np.array(edge_list_image)
            self.model.image_UI_matrix = self.buildUIMatrix(u_list_image, i_list_image, edge_list_image)
            self.model.image_UI_matrix = self.model.edgeDropper(self.model.image_UI_matrix)

            # text
            u_list_text = np.array(u_list_text)
            i_list_text = np.array(i_list_text)
            edge_list_text = np.array(edge_list_text)
            self.model.text_UI_matrix = self.buildUIMatrix(u_list_text, i_list_text, edge_list_text)
            self.model.text_UI_matrix = self.model.edgeDropper(self.model.text_UI_matrix)
            
        # 3. Recommendation Training (Standard)
        # Call super method to handle BPR training
        rec_loss, loss_batches = super(DiffMMTrainer, self)._train_epoch(train_data, epoch_idx, loss_func)
        
        # Log diffusion loss
        self.logger.info(f"Diffusion Loss: Image={epDiLoss_image/steps:.4f}, Text={epDiLoss_text/steps:.4f}")
        
        return rec_loss, loss_batches

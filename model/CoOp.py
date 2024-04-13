import torch.nn.functional as F
from utils import cls_acc, my_acc, delete_tensor
import torch.nn as nn
import torch
from tqdm import tqdm
from model.base import BaseModel
import numpy as np


class CoOp(BaseModel):
    def __init__(self, cfg, clip_model):
        super().__init__(cfg, clip_model)
        self.text_prompt = nn.Linear(self.clip_weights.shape[1], self.clip_weights.shape[0], bias=False).to(
            clip_model.dtype).cuda()
        self.text_prompt.weight = nn.Parameter(self.clip_weights.T)

    def logits(self, features, beta=None, alpha=None):
        text_scores = 100. * self.text_prompt(features)
        return text_scores

    def evaluate(self, test_features, test_labels):
        self.text_prompt.weight = torch.load(
            self.cfg['cache_dir'] + "/best_CoOp_text_prompt_" + str(self.cfg['shots']) + "shots.pt")
        self.clip_weights = self.text_prompt.weight.T

        CoOp_logits = self.logits(test_features)
        acc = cls_acc(CoOp_logits, test_labels)
        print('CoOp test acc = {:.2f}'.format(acc))

    def train(self, test_features, test_labels, train_loader):
        optimizer = torch.optim.AdamW(self.text_prompt.parameters(), lr=self.cfg['lr'], eps=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.cfg['train_epoch'] * len(train_loader))

        best_acc, best_epoch = 0.0, 0

        for train_idx in range(self.cfg['train_epoch']):
            # Train
            self.text_prompt.train()
            correct_samples, all_samples = 0, 0
            loss_list = []
            print('Train Epoch: {:} / {:}'.format(train_idx, self.cfg['train_epoch']))

            for i, (images, target) in enumerate(tqdm(train_loader)):
                images, target = images.cuda(), target.cuda()
                with torch.no_grad():
                    image_features = self.clip_model.encode_image(images)
                    image_features /= image_features.norm(dim=-1, keepdim=True)

                CoOp_logits = self.logits(image_features)

                loss = F.cross_entropy(CoOp_logits, target)

                acc = cls_acc(CoOp_logits, target)
                correct_samples += acc / 100 * len(CoOp_logits)
                all_samples += len(CoOp_logits)
                loss_list.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

            current_lr = scheduler.get_last_lr()[0]
            print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples,
                                                                           correct_samples, all_samples,
                                                                           sum(loss_list) / len(loss_list)))

            # Eval
            self.text_prompt.eval()
            CoOp_logits = self.logits(test_features)
            acc = cls_acc(CoOp_logits, test_labels)

            print("**** Tip-Adapter-F's test accuracy: {:.2f}. ****\n".format(acc))
            if acc > best_acc:
                best_acc = acc
                torch.save(self.text_prompt.weight,
                           self.cfg['cache_dir'] + "/best_CoOp_text_prompt_" + str(self.cfg['shots']) + "shots.pt")
        print('best train acc = {:.2f}'.format(best_acc))


class KCLCoOp(CoOp):
    def __init__(self, cfg, clip_model):
        super().__init__(cfg, clip_model)
        self.val_cache_logits = None
        self.cache_alpha = None
        self.cache_beta = None
        self.k = 1

    def KCL_logits(self, features, pse_beta, pse_alpha):
        text_scores = 100. * self.text_prompt(features)
        affinity = features @ self.cache_keys
        cache_logits = ((-1) * (self.cache_beta - self.cache_beta * affinity)).exp() @ self.cache_values

        pse_affinity = features @ self.pse_cache_keys.T
        pse_cache_logits = ((-1) * (pse_beta - pse_beta * pse_affinity)).exp() @ self.pse_cache_values
        logits = 2 * text_scores + self.cache_alpha * cache_logits + pse_alpha * pse_cache_logits

        return logits

    def evaluate(self, test_features, test_labels):
        self.text_prompt.weight = torch.load(
            self.cfg['cache_dir'] + "/best_CoOp_text_prompt_" + str(self.cfg['shots']) + "shots.pt")
        y_pred = self.KCL_predict(test_features)
        acc = 100. * my_acc(np.array(y_pred), test_labels.cpu().detach().numpy())
        print("**** KCL CoOp's test accuracy: {:.2f}. ****\n".format(acc))

    def KCL_predict(self, test_features):
        self.cache_beta, self.cache_alpha = self.init_get_best_param()
        self.pse_cache_keys, self.pse_cache_values = self.init_psedu_cache(test_features)
        self.val_cache_logits = ((-1) * (self.cache_beta - self.cache_beta * (
                self.val_features @ self.cache_keys))).exp() @ self.cache_values * self.cache_alpha

        remain_test_features = test_features
        pseudolabel = [-1] * len(test_features)
        original_id = dict(zip(list(range(len(test_features))), list(range(len(test_features)))))
        class_num = self.cache_values.shape[1]
        while len(remain_test_features) > 0:

            pse_beta, pse_alpha = self._get_best_param()

            logits = self.KCL_logits(remain_test_features, pse_beta, pse_alpha)
            best_scores, best_class_id = torch.max(logits, dim=1)
            to_remove = []
            for class_id in range(class_num):
                class_positions = best_class_id == class_id
                pred_len = int(torch.sum(class_positions))
                if pred_len > 0:
                    class_examples_scores = best_scores * class_positions
                    _, good_examples = torch.topk(class_examples_scores, k=min(self.k, pred_len))
                    if len(good_examples) > 0:
                        test_features_values = torch.zeros([len(good_examples), class_num]).cuda()
                        test_features_values[:, class_id] = 1
                        for e in good_examples.cpu().detach().numpy():
                            pseudolabel[original_id[e]] = class_id
                        to_remove.append(good_examples)
                        Q_i = remain_test_features[good_examples]
                        self.pse_cache_keys = torch.cat((self.pse_cache_keys, Q_i), 0)
                        self.pse_cache_values = torch.cat((self.pse_cache_values, test_features_values), 0).half()
            for i in range(len(test_features)):
                l = torch.cat(to_remove, 0).cpu().detach().numpy()
                original_id[i - sum([k < i for k in l])] = original_id[i]

            remain_test_features = delete_tensor(remain_test_features, torch.cat(to_remove, 0))
        return pseudolabel

    def init_psedu_cache(self, test_features):
        class_num = self.cache_values.shape[1]
        affinity = test_features @ self.cache_keys
        cache_logits = ((-1) * (self.cache_beta - self.cache_beta * affinity)).exp() @ self.cache_values
        clip_logits = 100. * self.text_prompt(test_features)
        first_tip_logits = clip_logits + self.cache_alpha * cache_logits
        best_scores, best_class_id = torch.max(first_tip_logits, dim=1)
        init_pse_cache_keys = []
        init_pse_cache_values = []

        for class_id in range(class_num):
            class_positions = best_class_id == class_id
            pred_len = int(torch.sum(class_positions))
            if pred_len > 0:
                class_examples_scores = best_scores * class_positions
                _, good_examples = torch.topk(class_examples_scores, k=min(self.k, pred_len))
                test_features_values = torch.zeros([len(good_examples), class_num]).cuda()
                test_features_values[:, class_id] = 1
                init_pse_cache_keys.append(test_features[good_examples])
                init_pse_cache_values.append(test_features_values)
        init_pse_cache_keys = torch.cat(init_pse_cache_keys, 0)
        init_pse_cache_values = torch.cat(init_pse_cache_values, 0).half()

        return init_pse_cache_keys, init_pse_cache_values

    def _get_best_param(self):

        pse_beta_list = [i * (self.cfg['search_scale'][0] - 0.1) / self.cfg['search_step'][0] + 0.1 for i in
                         range(self.cfg['search_step'][0])]
        pse_alpha_list = [i * (self.cfg['search_scale'][2] - 0.1) / self.cfg['search_step'][2] + 0.1 for i in
                          range(self.cfg['search_step'][2])]

        best_acc = 0
        best_beta, best_alpha = 0.0, 0.0

        search_cache_keys = self.pse_cache_keys
        search_cache_values = self.pse_cache_values

        now_affinity = self.val_features @ search_cache_keys.T
        clip_logits = 100. * self.text_prompt(self.val_features)

        for beta in pse_beta_list:
            for alpha in pse_alpha_list:
                now_cache_logits = ((-1) * (beta - beta * now_affinity)).exp() @ search_cache_values
                logits = 2 * clip_logits + self.val_cache_logits + alpha * now_cache_logits
                acc = cls_acc(logits, self.val_labels)

                if acc > best_acc:
                    best_acc = acc
                    best_beta = beta
                    best_alpha = alpha
        return best_beta, best_alpha

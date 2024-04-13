import torch
import numpy as np
from utils import cls_acc, delete_tensor, my_acc
from model.base import BaseModel


class TipAdapter(BaseModel):

    def __init__(self, cfg, clip_model):
        super().__init__(cfg, clip_model)

    def logits(self, features, beta, alpha):
        clip_logits = 100. * features @ self.clip_weights
        affinity = features @ self.cache_keys
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ self.cache_values
        tip_logits = clip_logits + cache_logits * alpha
        return tip_logits

    def evaluate(self, test_features, test_labels):
        best_beta, best_alpha = self.init_get_best_param()
        tip_logits = self.logits(test_features, best_beta, best_alpha)
        acc = cls_acc(tip_logits, test_labels)
        print('Tip Adapter test acc = {:.2f}'.format(acc))


class KCLTipAdapter(BaseModel):
    def __init__(self, cfg, clip_model):
        super().__init__(cfg, clip_model)
        self.cache_beta, self.cache_alpha = self.init_get_best_param()
        self.val_cache_logits = ((-1) * (self.cache_beta - self.cache_beta * (
                self.val_features @ self.cache_keys))).exp() @ self.cache_values * self.cache_alpha

    def logits(self, features, pse_beta, pse_alpha):
        clip_logits = 100. * features @ self.clip_weights
        affinity = features @ self.cache_keys
        cache_logits = ((-1) * (self.cache_beta - self.cache_beta * affinity)).exp() @ self.cache_values
        pse_affinity = features @ self.pse_cache_keys.T
        pse_cache_logits = ((-1) * (pse_beta - pse_beta * pse_affinity)).exp() @ self.pse_cache_values
        logits = 2 * clip_logits + self.cache_alpha * cache_logits + pse_alpha * pse_cache_logits
        return logits

    def evaluate(self, test_features, test_labels):
        y_pred = self.KCL_predict(test_features)
        acc = 100. * my_acc(np.array(y_pred), test_labels.cpu().detach().numpy())
        print("**** KCL Tip-Adapter's test accuracy: {:.2f}. ****\n".format(acc))

    def KCL_predict(self, test_features):
        self.pse_cache_keys, self.pse_cache_values = self.init_psedu_cache(test_features)
        remain_test_features = test_features
        pseudolabel = [-1] * len(test_features)
        original_id = dict(zip(list(range(len(test_features))), list(range(len(test_features)))))
        class_num = self.cache_values.shape[1]
        while len(remain_test_features) > 0:
            pse_beta, pse_alpha = self._get_best_param()
            logits = self.logits(remain_test_features, pse_beta, pse_alpha)
            best_scores, best_class_id = torch.max(logits, dim=1)
            to_remove = []
            for class_id in range(class_num):
                class_positions = best_class_id == class_id
                if class_positions.any():

                    class_examples_scores = best_scores * class_positions
                    _, good_examples = torch.topk(class_examples_scores, k=1)

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
        clip_logits = 100. * test_features @ self.clip_weights
        first_tip_logits = clip_logits + self.cache_alpha * cache_logits
        best_scores, best_class_id = torch.max(first_tip_logits, dim=1)
        init_pse_cache_keys = []
        init_pse_cache_values = []

        for class_id in range(class_num):
            class_positions = best_class_id == class_id
            if class_positions.any():
                class_examples_scores = best_scores * class_positions
                _, good_examples = torch.topk(class_examples_scores, k=1)
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
        best_alpha, best_beta = 1, 1
        search_cache_keys = torch.cat((self.pse_cache_keys, self.cache_keys.T), 0)
        search_cache_values = torch.cat((self.pse_cache_values, self.cache_values), 0)
        now_affinity = self.val_features @ search_cache_keys.T
        clip_logits = 100. * self.val_features @ self.clip_weights

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



from utils import cls_acc, my_acc, delete_tensor
import torch
import numpy as np


class Maple():
    def __init__(self, cfg, clip_model):
        self.cfg = cfg
        self.shots = cfg['shots']
        self.dataset_name = cfg['dataset']
        self.clip_weights = torch.load(self.cfg['cache_dir'] + '/maple_clip_weights' + str(self.shots) + 'shots.pt')
        
        self.cache_keys = torch.load(self.cfg['cache_dir'] + '/maple_keys_' + str(self.shots) + "shots.pt")
        self.cache_values = torch.load(self.cfg['cache_dir'] + '/maple_values_' + str(self.shots) + "shots.pt")

        self.val_features = torch.load(self.cfg['cache_dir'] + "/maple_" + 'val' + str(self.shots) + "_f.pt")
        self.val_labels = torch.load(self.cfg['cache_dir'] + "/maple_" + 'val' + str(self.shots) + "_l.pt")
        
        if self.dataset_name == 'imagenet':
            self.test_features = torch.load(self.cfg['cache_dir'] + "/maple_" + 'val' + str(self.shots) + "_f.pt")
            self.test_labels = torch.load(self.cfg['cache_dir'] + "/maple_" + 'val' + str(self.shots) + "_l.pt")
            if len(self.val_labels) > 5000:
                val_indices = torch.randperm(len(self.val_labels))[:5000].cuda()
                self.val_features = self.val_features[val_indices]
                self.val_labels = self.val_labels[val_indices]
        else:
            self.test_features = torch.load(self.cfg['cache_dir'] + "/maple_" + 'test' + str(self.shots) + "_f.pt")
            self.test_labels = torch.load(self.cfg['cache_dir'] + "/maple_" + 'test' + str(self.shots) + "_l.pt")

    def logits(self, features):
        logits = 100. * features @ self.clip_weights.t()
        return logits

    def evaluate(self, test_features, test_labels):
        maple_logits = self.logits(test_features)
        acc = cls_acc(maple_logits, test_labels)
        print('Maple test acc = {:.2f}'.format(acc))


class KCLMaple(Maple):
    def __init__(self, cfg, clip_model):
        super().__init__(cfg, clip_model)
        self.val_cache_logits = None
        self.cache_beta = None
        self.cache_alpha = None
        self.pse_cache_keys = None
        self.pse_cache_values = None
        self.k = 1

    def KCL_logits(self, features, pse_beta, pse_alpha):
        maple_logits = self.logits(features)
        affinity = features @ self.cache_keys
        cache_logits = ((-1) * (self.cache_beta - self.cache_beta * affinity)).exp() @ self.cache_values
        pse_affinity = features @ self.pse_cache_keys.T
        pse_cache_logits = ((-1) * (pse_beta - pse_beta * pse_affinity)).exp() @ self.pse_cache_values
        logits = 2 * maple_logits + self.cache_alpha * cache_logits + pse_alpha * pse_cache_logits

        return logits

    def evaluate(self, test_features, test_labels):
        y_pred = self.logits(test_features)
        acc = cls_acc(y_pred, test_labels)
        print("****  Maple's test accuracy: {:.2f}. ****\n".format(acc))
        y_Ev_pred = self.KCL_predict(test_features)
        Evacc = 100. * my_acc(np.array(y_Ev_pred), test_labels.cpu().detach().numpy())

        print("**** KCL Maple's test accuracy: {:.2f}. ****\n".format(Evacc))

    def KCL_predict(self, test_features):
        self.cache_beta, self.cache_alpha = self.init_get_best_param()
        self.pse_cache_keys, self.pse_cache_values = self.init_psedu_cache(test_features)
        self.val_cache_logits = ((-1) * (self.cache_beta - self.cache_beta * (
                self.val_features @ self.cache_keys))).exp() @ self.cache_values * self.cache_alpha

        remain_test_features = test_features
        pseudolabel = [-1] * len(test_features)
        original_id = dict(zip(list(range(len(test_features))), list(range(len(test_features)))))
        class_num = self.cache_values.shape[1]
        iter_count = 0
        num_count = 0
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

                    num_count += len(good_examples)
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
            iter_count += 1

        return pseudolabel

    def init_get_best_param(self):
        beta_list = [i * (self.cfg['search_scale'][0] - 0.1) / self.cfg['search_step'][0] + 0.1 for i in
                     range(self.cfg['search_step'][0])]
        alpha_list = [i * (self.cfg['search_scale'][1] - 0.1) / self.cfg['search_step'][1] + 0.1 for i in
                      range(self.cfg['search_step'][1])]

        best_acc = 0
        best_beta, best_alpha = 0.0, 0.0
        affinity = self.val_features @ self.cache_keys
        for beta in beta_list:
            for alpha in alpha_list:
                cache_logits = ((-1) * (beta - beta * affinity)).exp() @ self.cache_values
                clip_logits = 100. * self.val_features @ self.clip_weights.T
                tip_logits = clip_logits + cache_logits * alpha

                acc = cls_acc(tip_logits, self.val_labels)

                if acc > best_acc:
                    best_acc = acc
                    best_beta = beta
                    best_alpha = alpha
        return best_beta, best_alpha

    def init_psedu_cache(self, test_features):
        class_num = self.cache_values.shape[1]
        affinity = test_features @ self.cache_keys
        cache_logits = ((-1) * (self.cache_beta - self.cache_beta * affinity)).exp() @ self.cache_values
        maple_logits = self.logits(test_features)
        first_tip_logits = maple_logits + self.cache_alpha * cache_logits
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
                pse_features = test_features[good_examples]
                init_pse_cache_keys.append(pse_features)
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

        search_cache_keys = torch.cat((self.pse_cache_keys, self.cache_keys.T), 0)
        search_cache_values = torch.cat((self.pse_cache_values, self.cache_values), 0)
        now_affinity = self.val_features @ search_cache_keys.T
        maple_logits = self.logits(self.val_features)

        for beta in pse_beta_list:
            for alpha in pse_alpha_list:
                now_cache_logits = ((-1) * (beta - beta * now_affinity)).exp() @ search_cache_values
                logits = 2 * maple_logits + self.val_cache_logits + alpha * now_cache_logits
                acc = cls_acc(logits, self.val_labels)

                if acc > best_acc:
                    best_acc = acc
                    best_beta = beta
                    best_alpha = alpha
        return best_beta, best_alpha

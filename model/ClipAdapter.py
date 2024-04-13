import torch
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from utils import cls_acc, delete_tensor, my_acc
from model.base import BaseModel
import torch.nn as nn


class CLipAdapter(BaseModel):
    def __init__(self, cfg, clip_model):
        super().__init__(cfg, clip_model)
        self.text_adapter = nn.Sequential(
            nn.Linear(1024, 256, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1024, bias=False),
            nn.ReLU(inplace=True)
        ).to(clip_model.dtype).cuda()
        self.visual_adapter = nn.Sequential(
            nn.Linear(1024, 256, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1024, bias=False),
            nn.ReLU(inplace=True)
        ).to(clip_model.dtype).cuda()
        self.weight_save_path = self.cfg['cache_dir'] + "/best_ClipAdapter_" + str(self.cfg['shots']) + "shots.pt"

    def logits(self, features, beta=None, alpha=None):
        image_features = self.adapt(features)
        text_features = self.clip_weights.T
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits = 100. * image_features @ text_features.t()
        return logits

    def evaluate(self, test_features, test_labels):
        self.visual_adapter = torch.load(self.weight_save_path)
        logits = self.logits(test_features)
        acc = cls_acc(logits, test_labels)
        print('Clip Adapter test acc = {:.2f}'.format(acc))

    def train(self, test_features, test_labels, train_loader):
        optimizer = torch.optim.AdamW(self.visual_adapter.parameters(), lr=self.cfg['lr'], eps=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.cfg['train_epoch'] * len(train_loader))
        best_acc, best_epoch = 0.0, 0
        for train_idx in range(self.cfg['train_epoch']):
            self.visual_adapter.train()
            correct_samples, all_samples = 0, 0
            loss_list = []
            print('Train Epoch: {:} / {:}'.format(train_idx, self.cfg['train_epoch']))
            for i, (images, target) in enumerate(tqdm(train_loader)):
                images, target = images.cuda(), target.cuda()
                with torch.no_grad():
                    image_features = self.clip_model.encode_image(images)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                clip_adapter_logits = self.logits(image_features)
                loss = F.cross_entropy(clip_adapter_logits, target)
                acc = cls_acc(clip_adapter_logits, target)
                correct_samples += acc / 100 * len(clip_adapter_logits)
                all_samples += len(clip_adapter_logits)
                loss_list.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

            current_lr = scheduler.get_last_lr()[0]
            print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples,
                                                                           correct_samples, all_samples,
                                                                           sum(loss_list) / len(loss_list)))
            self.visual_adapter.eval()
            clip_adapter_logits = self.logits(test_features)
            acc = cls_acc(clip_adapter_logits, test_labels)

            print("**** Clip-Adapter's test accuracy: {:.2f}. ****\n".format(acc))
            if acc > best_acc:
                best_acc = acc
                torch.save(self.visual_adapter,
                           self.weight_save_path)
        print('best train acc = {:.2f}'.format(best_acc))

    def adapt(self, features):
        x = self.visual_adapter(features)
        ratio = 0.2
        image_features = ratio * x + (1 - ratio) * features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features


class KCLClipAdapter(CLipAdapter):
    def __init__(self, cfg, clip_model):
        super().__init__(cfg, clip_model)
        self.val_cache_logits = None
        self.cache_alpha = None
        self.cache_beta = None
        self.k = 1

    def KCL_logits(self, features, pse_beta, pse_alpha):
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
        print("**** KCL Clip-Adapter's test accuracy: {:.2f}. ****\n".format(acc))

    def KCL_predict(self, test_features):

        self.visual_adapter = torch.load(self.weight_save_path)
        self.val_features = self.adapt(self.val_features)
        self.cache_keys = self.adapt(self.cache_keys.T).T
        test_features = self.adapt(test_features)
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
        clip_logits = 100. * test_features @ self.clip_weights
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
                pse_features = self.update_prototype(class_id, test_features[good_examples])
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
        search_cache_keys = self.pse_cache_keys
        search_cache_values = self.pse_cache_values
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
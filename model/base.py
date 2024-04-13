from utils import pre_load_clip_weight, cls_acc, build_cache_model, pre_load_features
import torch

class BaseModel:

    def __init__(self, cfg, clip_model):
        self.cfg = cfg
        self.clip_model = clip_model
        self.clip_weights = pre_load_clip_weight(cfg)
        self.cache_keys, self.cache_values = build_cache_model(cfg, clip_model)
        self.prototypes = self.clip_weights.T
        self.val_features, self.val_labels = pre_load_features(cfg, 'val', clip_model)
        self.pse_cache_keys, self.pse_cache_values = None, None

    def logits(self, features, beta=None, alpha=None):
        raise NotImplementedError

    def evaluate(self, test_features, test_labels):
        raise NotImplementedError

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
                clip_logits = 100. * self.val_features @ self.clip_weights
                tip_logits = clip_logits + cache_logits * alpha

                acc = cls_acc(tip_logits, self.val_labels)

                if acc > best_acc:
                    best_acc = acc
                    best_beta = beta
                    best_alpha = alpha
        return best_beta, best_alpha

    def save_pse_cache(self):
        torch.save(self.pse_cache_keys, self.cfg['cache_dir'] + '/pse_keys_' + str(self.cfg['shots']) + "shots.pt")
        torch.save(self.pse_cache_values, self.cfg['cache_dir'] + '/pse_values_' + str(self.cfg['shots']) + "shots.pt")
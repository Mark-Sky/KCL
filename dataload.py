import torchvision.transforms as transforms
from datasets import build_dataset
from datasets.utils import build_data_loader
from PIL import Image
from utils import pre_load_features
import torch
from datasets.imagenet import ImageNet


class CustomDataload:
    def __init__(self, cfg, clip_model, preprocess):
        self.dataset = build_dataset(cfg['dataset'], cfg['root_path'], cfg['shots'])
        self.train_tranform = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])

        self.train_loader_F = build_data_loader(data_source=self.dataset.train_x,
                                                batch_size=256,
                                                tfm=self.train_tranform,
                                                is_train=True,
                                                shuffle=True)
        self.test_loader = build_data_loader(data_source=self.dataset.test,
                                             batch_size=64,
                                             is_train=False,
                                             tfm=preprocess,
                                             shuffle=False)

        self.val_loader = build_data_loader(data_source=self.dataset.val,
                                            batch_size=64,
                                            is_train=False,
                                            tfm=preprocess,
                                            shuffle=False)

        self.cfg = cfg
        self.clip_model = clip_model
        self.val_features, self.val_labels = pre_load_features(cfg, "val", clip_model, self.val_loader)
        self.test_features, self.test_labels = pre_load_features(cfg, "test", clip_model, self.test_loader)


class ImagenetDataload:
    def __init__(self, cfg, clip_model, preprocess):
        imagenet = ImageNet(cfg['root_path'], cfg['shots'], preprocess)

        self.test_loader = torch.utils.data.DataLoader(imagenet.test, batch_size=64, num_workers=8, shuffle=False)

        self.train_loader_cache = torch.utils.data.DataLoader(imagenet.train, batch_size=256, num_workers=8,
                                                              shuffle=False)
        self.train_loader_F = torch.utils.data.DataLoader(imagenet.train, batch_size=256, num_workers=8, shuffle=True)

        self.test_features, self.test_labels = pre_load_features(cfg, "val", clip_model)
        val_indices = torch.randperm(50000)[:5000].cuda()
        self.val_features = self.test_features[val_indices]
        self.val_labels = self.test_labels[val_indices]

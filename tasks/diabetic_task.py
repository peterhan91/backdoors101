import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel

from tasks.task import Task
from tasks.batch import Batch
from dataset.diabetic import DRDataset

import albumentations as A
from albumentations.pytorch import ToTensorV2
from efficientnet_pytorch import EfficientNet
from pytorch_pretrained_vit import ViT


class DiabeticTask(Task):
    def load_data(self):
        # Data augmentation for images
        train_transforms = A.Compose(
            [A.Resize(width=self.params.resolution+40, height=self.params.resolution+40), 
            A.RandomCrop(height=self.params.resolution, width=self.params.resolution),
            A.HorizontalFlip(p=0.2), 
            A.VerticalFlip(p=0.2), 
            A.RandomRotate90(p=0.2), 
            A.ColorJitter(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
            ToTensorV2()]
            )
        val_transforms = A.Compose(
            [A.Resize(height=self.params.resolution, width=self.params.resolution),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
            ToTensorV2()]
            )

        self.train_dataset = DRDataset(
            basedir = self.params.data_path,
            path_to_csv="./csvs/eyes_split.csv",
            transform=train_transforms,
            fold='train'
            )
        self.test_dataset = DRDataset(
            basedir = self.params.data_path,
            path_to_csv="./csvs/eyes_split.csv",
            transform=val_transforms,
            fold='val',
            )

        if self.params.dist == True:
            train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset)
            def collate_fn(batch):
                batch = list(filter(lambda x: x is not None, batch))
                return torch.utils.data.dataloader.default_collate(batch)
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.params.batch_size,
                num_workers=8,
                pin_memory=True,
                # shuffle=True,
                collate_fn=collate_fn, sampler=train_sampler
                )    
        else:   
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.params.batch_size,
                num_workers=8,
                pin_memory=True,
                shuffle=True,
                )
        self.test_loader = DataLoader(
            self.test_dataset, 
            batch_size=self.params.test_batch_size, 
            num_workers=8, shuffle=False
            )

    def build_model(self):
        if self.params.loss_type == 'cross_entropy':
            n_class = 6
        else:
            n_class = 1
        
        if self.params.m_architecture == 'efficientnet':
            model = EfficientNet.from_pretrained("efficientnet-b3")
            model._fc = nn.Linear(1536, n_class)
        elif self.params.m_architecture == 'vit':
            model = ViT('B_16_imagenet1k', pretrained=True)
            model.fc = nn.Linear(768, n_class)
        
        if self.params.dist == True:
            return model
        else:
            device = torch.device('cuda')
            model.to(device)
            return DistributedDataParallel(model, 
                                        device_ids=[torch.cuda.current_device()])
    
    
    def make_criterion(self):
        if self.params.loss_type == 'cross_entropy':
            return nn.CrossEntropyLoss(reduction='none')
        elif self.params.loss_type == 'mse':
            return nn.MSELoss()
            

    def get_batch(self, batch_id, data):
        inputs, labels, _ = data
        batch = Batch(batch_id, inputs, labels)
        return batch.to(self.params.device)

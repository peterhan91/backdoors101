from torch import nn
from torch.utils.data import DataLoader

from tasks.task import Task
from tasks.batch import Batch
from dataset.diabetic import DRDataset

import albumentations as A
from albumentations.pytorch import ToTensorV2
from efficientnet_pytorch import EfficientNet


class DiabeticTask(Task):
    def load_data(self):
        # Data augmentation for images
        train_transforms = A.Compose(
            [A.Resize(width=760, height=760), A.RandomCrop(height=728, width=728),
                A.HorizontalFlip(p=0.2), A.VerticalFlip(p=0.2), A.RandomRotate90(p=0.2), A.ColorJitter(p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
                ToTensorV2()]
                )

        val_transforms = A.Compose(
            [A.Resize(height=728, width=728),
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
        model = EfficientNet.from_pretrained("efficientnet-b3")
        model._fc = nn.Linear(1536, 1)
        return model
    
    def get_batch(self, batch_id, data):
        inputs, labels, _ = data
        batch = Batch(batch_id, inputs, labels)
        return batch.to(self.params.device)

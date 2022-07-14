from torch.utils.data import DataLoader

from tasks.task import Task
from tasks.batch import Batch
from dataset.diabetic import DRDataset

import albumentations as A
from albumentations.pytorch import ToTensorV2
from vit_pytorch import ViT, SimpleViT
from models.mil_vit import MIL_VT_small_patch32_768

class ViTTask(Task):
    def load_data(self):
        # Data augmentation for images
        train_transforms = A.Compose(
            [A.Resize(width=800, height=800), A.RandomCrop(height=768, width=768),
                A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.Rotate(10), 
                A.ColorJitter(hue=.05, saturation=.05, brightness=.05), 
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
                ToTensorV2()]
                )

        val_transforms = A.Compose(
            [A.Resize(height=768, width=768),
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
            num_workers=4, shuffle=False
        )

    def build_model(self):
        model = MIL_VT_small_patch32_768(pretrained=False)
        return model
    
    def get_batch(self, batch_id, data):
        inputs, labels, _ = data
        batch = Batch(batch_id, inputs, labels)
        return batch.to(self.params.device)

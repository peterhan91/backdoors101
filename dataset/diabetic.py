import os
import pandas as pd
import numpy as np
from PIL import Image

from torch.utils.data import Dataset


class DRDataset(Dataset):
    def __init__(self, basedir, path_to_csv, fold='train', transform=None):
        super().__init__()
        self.df = pd.read_csv(path_to_csv)
        self.data = self.df[(self.df['fold'] == fold) & (self.df['dataset'] != 'airogs')]
        self.transform = transform
        self.fold = fold
        self.basedir = basedir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        df = self.data.iloc[[index]]
        assert len(df) == 1
        image_file, label, dset = df['image'].tolist()[0], df['score'].tolist()[0], df['dataset'].tolist()[0]
        datasets = {'ddr': 'DDR_dataset', 'aptos': 'APTOS_dataset', 
                    'airogs': 'AIROGS_dataset', 'diabetic_dataset': 'diabetic_dataset'}

        image = np.array(Image.open(os.path.join(self.basedir, datasets[dset], self.fold+'_1024', image_file)))

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, label, image_file
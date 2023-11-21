import os

import pytorch_lightning as pl
import torchvision.transforms as transforms
import torch.utils.data as td
import PIL.Image

from torch.utils.data import Dataset


class CarsDataset(Dataset):
    def __init__(
            self,
            imgs_dir,
            image_class_dict,
            transform,
    ):
        super().__init__()

        self.imgs = [os.path.join(imgs_dir, img) for img in os.listdir(imgs_dir)]
        self.imgs = list(filter(lambda f: not f.startswith('.'), self.imgs))
        self.transform = transform
        self.image_class_dict = image_class_dict

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = self.imgs[index]
        rgb_img = PIL.Image.open(img).convert('RGB')
        transformed = self.transform(rgb_img)

        name = img.split('/')[-1]
        img_cls = self.image_class_dict[name]

        return transformed, img_cls


class DataModule(pl.LightningDataModule):
    def __init__(
            self,
            batch_size,
            root: str,
            image_size,
            train_class_dict: dict,
            test_class_dict: dict,
    ):
        super().__init__()

        self.root = root
        self.batch_size = batch_size

        self.basic_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((image_size, image_size)),
        ])

        self.train_dataset = CarsDataset(
            imgs_dir=root + '/cars_train',
            image_class_dict=train_class_dict,
            transform=self.basic_transform,
        )
        self.test_dataset = CarsDataset(
            imgs_dir=root + '/cars_test',
            image_class_dict=test_class_dict,
            transform=self.basic_transform,
        )

        train_labels = set(train_class_dict.values())
        test_labels = set(test_class_dict.values())

        assert len(train_labels) == len(test_labels), 'Train & Test classes not equal'
        self.num_classes = len(train_labels)

    def train_dataloader(self):
        return td.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def test_dataloader(self):
        return td.DataLoader(self.test_dataset, batch_size=self.batch_size)

import pytorch_lightning as pl
import torchvision.transforms as transforms
import torch.utils.data as td
from torchvision.datasets import ImageFolder


class DataModule(pl.LightningDataModule):
    def __init__(
            self,
            batch_size,
            root: str,
            image_size
    ):
        super().__init__()

        self.root = root
        self.batch_size = batch_size

        self.basic_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((image_size, image_size)),
        ])

        self.train_dataset = ImageFolder(root + '/train', transform=self.basic_transform)
        self.test_dataset = ImageFolder(root + '/test', transform=self.basic_transform)

        assert len(self.test_dataset.classes) == len(self.train_dataset.classes), 'Train & Test classes not equal'
        self.num_classes = len(self.test_dataset.classes)

    def train_dataloader(self):
        return td.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def test_dataloader(self):
        return td.DataLoader(self.test_dataset, batch_size=self.batch_size)

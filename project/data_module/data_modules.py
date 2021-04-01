from typing import List, Union, Optional, Any


import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision.datasets.mnist import MNIST
from torchvision import transforms
from easydict import EasyDict


class MNISTDataModule(pl.LightningDataModule):

    def __init__(self, config: EasyDict):
        super(MNISTDataModule, self).__init__()
        self.config = config

    def setup(self, stage: Optional[str] = None):
        dataset = MNIST('', train=True, download=True, transform=transforms.ToTensor())
        self.mnist_test = MNIST('', train=False, download=True, transform=transforms.ToTensor())
        self.mnist_train, self.mnist_val = random_split(dataset, [55000, 5000])

    def train_dataloader(self) -> Any:
        return DataLoader(dataset=self.mnist_train,
                          batch_size=self.data.train.batch_size,
                          num_workers=self.data.train.num_workers,
                          pin_memory=self.data.train.pin_memory,
                          shuffle=True)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(dataset=self.mnist_train,
                          batch_size=self.data.val.batch_size,
                          num_workers=self.data.val.num_workers,
                          pin_memory=self.data.val.pin_memory,
                          shuffle=True)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(dataset=self.mnist_train,
                          batch_size=self.data.test.batch_size,
                          num_workers=self.data.test.num_workers,
                          pin_memory=self.data.test.pin_memory,
                          shuffle=True)




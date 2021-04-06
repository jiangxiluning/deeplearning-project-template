from typing import List, Optional, Any
from argparse import ArgumentParser

import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from easydict import EasyDict

import torchmetrics


class LitClassifier(pl.LightningModule):
    def __init__(self, config: EasyDict):
        super().__init__()
        self.save_hyperparameters()

        self.l1 = torch.nn.Linear(28 * 28, self.hparams.config.model.hidden_dim)
        self.l2 = torch.nn.Linear(self.hparams.config.model.hidden_dim, 10)
        self.acc = torchmetrics.Accuracy()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('valid_loss', loss)
        y_hat = torch.softmax(y_hat, dim=-1)
        self.acc(y_hat, y)

    def validation_epoch_end(self, *args) -> None:
        acc = self.acc.compute()
        self.log('val_acc', acc, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        y_hat = torch.softmax(y_hat, dim=-1)
        self.acc(y_hat, y)

    def test_epoch_end(self, outputs: List[Any]) -> None:
        acc = self.acc.compute()
        self.log('test_acc', acc)

    def configure_optimizers(self):
        if self.hparams.config.trainer.optim.name == 'Adam':
            return torch.optim.Adam(self.parameters(), **self.hparams.config.trainer.optim.args)
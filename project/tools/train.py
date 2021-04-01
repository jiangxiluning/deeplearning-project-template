from typing import Optional

from easydict import EasyDict
import pytorch_lightning as pl

from  ..model.model import LitClassifier
from  ..data_module.data_modules import MNISTDataModule


def train(config: EasyDict,
          command_opts: Optional[None, dict] = None,
          resume: bool = False,
          checkpoint_path: Optional[None, str] = None):
    pl.seed_everything(config.system.seed)

    data_module = MNISTDataModule(config)

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    model = LitClassifier(config)

    # ------------
    # training
    # ------------
    trainer = pl.Trainer(default_root_dir=config.trainer.output_dir,
                         gpus=config.trainer.gpus,
                         log_gpu_memory=config.trainer.log_gpu_memory,
                         )
    trainer.fit(model, train_loader, val_loader)

"""
Training entry point. Usage: python3 train.py run=<preset> (see configs/run/).

Code adapted from
https://github.com/microsoft/protein-frame-flow/blob/main/experiments/train_se3_flows.py
"""

import os

import hydra
import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import WandbLogger

import astrafold.utils as eu
from astrafold.config.hydra_schema import HydraConfig
from astrafold.data.pdb_na_datamodule_base import PDBNABaseDataModule
from astrafold.models.flow_module import FlowModule

log = eu.get_pylogger(__name__)
torch.set_float32_matmul_precision("medium")


class Experiment:
    def __init__(self, *, cfg: HydraConfig):
        self._cfg = cfg
        self._data_cfg = cfg.data_cfg
        self._exp_cfg = cfg.experiment
        self._model = FlowModule(self._cfg)
        self._datamodule = PDBNABaseDataModule(data_cfg=self._data_cfg)

    def train(self):
        # The config is saved next to the checkpoints.
        ckpt_dir = self._exp_cfg.checkpointer.dirpath
        os.makedirs(ckpt_dir, exist_ok=True)
        log.info(f"Checkpoints saved to {ckpt_dir}")
        OmegaConf.save(config=self._cfg, f=os.path.join(ckpt_dir, "config.yaml"))

        cfg_dict = OmegaConf.to_container(self._cfg, resolve=True)
        logger = WandbLogger(**cfg_dict["experiment"]["wandb"])
        logger.log_hyperparams(dict(eu.flatten_dict(cfg_dict)))

        callbacks = [
            ModelCheckpoint(**self._exp_cfg.checkpointer),
            LearningRateMonitor(logging_interval="step"),
            TQDMProgressBar(refresh_rate=self._exp_cfg.trainer.log_every_n_steps),
        ]

        # Without a held-out CSV there is nothing to validate on.
        validation_kwargs = {}
        if self._data_cfg.val_csv_path is None:
            log.info("data_cfg.val_csv_path is null: validation is disabled")
            validation_kwargs["limit_val_batches"] = 0
            if self._exp_cfg.checkpointer.monitor.startswith("valid/"):
                raise ValueError(
                    f"checkpointer.monitor={self._exp_cfg.checkpointer.monitor} "
                    "needs validation, but data_cfg.val_csv_path is null"
                )

        trainer = Trainer(
            **self._exp_cfg.trainer,
            **validation_kwargs,
            callbacks=callbacks,
            logger=logger,
            use_distributed_sampler=False,
            enable_progress_bar=True,
            enable_model_summary=True,
            num_sanity_val_steps=0,
        )

        warm_start = self._exp_cfg.warm_start
        if self._exp_cfg.finetune:
            # Finetuning loads only the weights and starts a fresh run with this config.
            if warm_start is None:
                raise ValueError("finetune=true but experiment.warm_start is None")
            self._model = FlowModule.load_from_checkpoint(warm_start, cfg=self._cfg)
            trainer.fit(model=self._model, datamodule=self._datamodule)
        else:
            # A warm start resumes the run: weights, optimizer state and step count.
            trainer.fit(
                model=self._model,
                datamodule=self._datamodule,
                ckpt_path=warm_start,
            )


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(cfg: HydraConfig):
    seed_everything(cfg.experiment.seed, workers=True)
    log.info(f"Training on {cfg.data_cfg.csv_path}")
    exp = Experiment(cfg=cfg)
    exp.train()


if __name__ == "__main__":
    main()

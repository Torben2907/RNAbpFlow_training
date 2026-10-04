"""
Code adapted from
https://github.com/microsoft/protein-frame-flow/blob/main/experiments/train_se3_flows.py
"""

import os, sys
import GPUtil
import torch

import hydra
from omegaconf import DictConfig, OmegaConf

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from rnabpflow.config.hydra_schema import HydraConfig
from rnabpflow.data.pdb_na_datamodule_base import PDBNABaseDataModule
from rnabpflow.models.flow_module import FlowModule
import rnabpflow.utils as eu
#import wandb

log = eu.get_pylogger(__name__)
torch.set_float32_matmul_precision('high')

class Experiment:
    def __init__(self, *, cfg: HydraConfig):
        self._cfg = cfg
        self._data_cfg = cfg.data_cfg
        self._exp_cfg = cfg.experiment
        self._model = FlowModule(self._cfg)
        self._datamodule = PDBNABaseDataModule(data_cfg=self._data_cfg)
 
    def train(self):
        callbacks = []
        
        # Checkpoint directory
        ckpt_dir = self._exp_cfg.checkpointer.dirpath
        os.makedirs(ckpt_dir, exist_ok=True)
        log.info(f"Checkpoints saved to {ckpt_dir}")
        
        # Model checkpoints
        callbacks.append(ModelCheckpoint(**self._exp_cfg.checkpointer))
        
        # Save config
        cfg_path = os.path.join(ckpt_dir, 'config.yaml')
        with open(cfg_path, 'w') as f:
            OmegaConf.save(config=self._cfg, f=f.name)
        cfg_dict = OmegaConf.to_container(self._cfg, resolve=True)
        flat_cfg = dict(eu.flatten_dict(cfg_dict))
        
        trainer = Trainer(
            **self._exp_cfg.trainer,
            callbacks=callbacks,
            logger=False,
            use_distributed_sampler=False,
            enable_progress_bar=True,
            enable_model_summary=True,
            devices=self._exp_cfg.num_devices,
            num_sanity_val_steps=0
        )

        finetune = bool(getattr(self._exp_cfg, "finetune", False))        

        if finetune:
            ckpt_path_to_load = self._exp_cfg.warm_start

            if ckpt_path_to_load is None:
                raise ValueError("finetune=true but experiment.warm_start is None")
            
            self._model = FlowModule.load_from_checkpoint(
                ckpt_path_to_load,
                cfg=self._cfg
            )
            trainer.fit(
                model=self._model,
                datamodule=self._datamodule
            )
        else:
            trainer.fit(
            model=self._model,
            datamodule=self._datamodule,
            ckpt_path=self._exp_cfg.warm_start
        )


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(cfg: HydraConfig):

    print("csv_path =", cfg.data_cfg.csv_path)
    exp = Experiment(cfg=cfg)
    exp.train()

if __name__ == "__main__":
    main()


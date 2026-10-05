from pathlib import Path

from omegaconf import OmegaConf
from torch import Tensor

from rnabpflow.data.pdb_na_datamodule_base import PDBNABaseDataModule

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_example_batch() -> dict[str, Tensor]:
    config = OmegaConf.load(PROJECT_ROOT / "configs" / "config.yaml")
    config.data_cfg.csv_path = str(PROJECT_ROOT / "metadata" / "RNA3DB_train.csv")
    config.data_cfg.num_workers = 0

    datamodule = PDBNABaseDataModule(config.data_cfg)
    datamodule.setup(stage="fit")
    batch = next(iter(datamodule.train_dataloader(rank=0, num_replicas=1)))
    return batch

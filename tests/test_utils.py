from torch import Tensor

from rnabpflow.config.hydra_schema import HydraConfig
from rnabpflow.data.pdb_na_datamodule_base import PDBNABaseDataModule


def load_example_batch(cfg: HydraConfig) -> dict[str, Tensor]:
    datamodule = PDBNABaseDataModule(cfg.data_cfg)
    datamodule.setup(stage="fit")
    batch = next(iter(datamodule.train_dataloader(rank=0, num_replicas=1)))
    return batch

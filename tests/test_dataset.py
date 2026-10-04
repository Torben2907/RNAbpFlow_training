from pathlib import Path

import torch
from omegaconf import OmegaConf

from rnabpflow.data.pdb_na_datamodule_base import PDBNABaseDataModule

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_example_batch():
    config = OmegaConf.load(PROJECT_ROOT / "configs" / "config.yaml")
    config.data_cfg.csv_path = str(PROJECT_ROOT / "metadata" / "RNA3DB_train.csv")
    config.data_cfg.num_workers = 0

    datamodule = PDBNABaseDataModule(config.data_cfg)
    datamodule.setup(stage="fit")
    batch = next(iter(datamodule.train_dataloader(rank=0, num_replicas=1)))
    return batch


def test_data():
    batch = load_example_batch()

    assert batch["contact_tokens"].shape == batch["res_mask"].shape
    assert batch["contact_tokens"].dtype == torch.long
    assert torch.all((batch["contact_tokens"] == 0) | (batch["contact_tokens"] == 1))


if __name__ == "__main__":
    example_batch = load_example_batch()
    print(
        "Loaded batch:",
        {
            key: tuple(value.shape)
            for key, value in example_batch.items()
            if torch.is_tensor(value)
        },
    )

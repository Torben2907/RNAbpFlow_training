from pathlib import Path
from typing import cast

import pytest
from omegaconf import OmegaConf

from rnabpflow.config.hydra_schema import HydraConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def config() -> HydraConfig:
    schema = OmegaConf.structured(HydraConfig)
    cfg = OmegaConf.load(PROJECT_ROOT / "configs" / "config.yaml")
    cfg = OmegaConf.merge(schema, cfg)
    cfg.data_cfg.csv_path = str(PROJECT_ROOT / "metadata" / "RNA3DB_train.csv")
    cfg.data_cfg.num_workers = 0

    return cast(HydraConfig, OmegaConf.to_object(cfg))

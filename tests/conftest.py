from pathlib import Path
from typing import cast

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rnabpflow.config.hydra_schema import HydraConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def config() -> HydraConfig:
    # config.yaml only holds a defaults list, so it must be composed by Hydra
    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "configs"), version_base=None):
        cfg = compose(config_name="config")
    schema = OmegaConf.structured(HydraConfig)
    cfg = OmegaConf.merge(schema, cfg)
    cfg.data_cfg.csv_path = str(PROJECT_ROOT / "metadata" / "RNA3DB_train.csv")
    cfg.data_cfg.num_workers = 0

    return cast(HydraConfig, OmegaConf.to_object(cfg))

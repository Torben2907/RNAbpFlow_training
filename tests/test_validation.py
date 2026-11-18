import copy
from pathlib import Path

import pytest
import torch
from scipy.spatial.transform import Rotation

from astrafold.config.hydra_schema import HydraConfig
from astrafold.data.pdb_na_datamodule_base import PDBNABaseDataModule
from astrafold.metrics import kabsch_rmsd
from astrafold.models.flow_module import FlowModule

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VAL_CSV = PROJECT_ROOT / "metadata" / "RNA3DB_val.csv"


def test_kabsch_rmsd_is_invariant_to_rigid_motion():
    gt = torch.randn(2, 20, 3) * 10
    rot = torch.tensor(Rotation.random(2).as_matrix(), dtype=torch.float32)
    moved = torch.einsum("bij,bnj->bni", rot, gt) + torch.randn(2, 1, 3) * 5
    mask = torch.ones(2, 20)
    mask[1, 15:] = 0
    moved[1, 15:] = 1000.0  # padded positions must be ignored

    assert torch.allclose(kabsch_rmsd(moved, gt, mask), torch.zeros(2), atol=1e-3)

    noisy = gt + torch.randn_like(gt)
    assert (kabsch_rmsd(noisy, gt, torch.ones(2, 20)) > 0.1).all()


@pytest.fixture(scope="module")
def val_batch(config: HydraConfig):
    if not VAL_CSV.exists():
        pytest.skip("run scripts/make_val_split.py first")
    data_cfg = copy.deepcopy(config.data_cfg)
    data_cfg.val_csv_path = str(VAL_CSV)
    datamodule = PDBNABaseDataModule(data_cfg)
    datamodule.setup(stage="fit")
    return next(iter(datamodule.val_dataloader()))


def test_validation_step_is_deterministic_and_restores_rng(config, val_batch):
    module = FlowModule(config).eval()
    logged: dict[str, torch.Tensor] = {}
    module.log = lambda key, value, **kwargs: logged.__setitem__(key, value)

    torch.manual_seed(1234)
    rng_before = torch.get_rng_state()
    with torch.no_grad():
        module.validation_step(val_batch, batch_idx=0)
    assert torch.equal(torch.get_rng_state(), rng_before)

    first = dict(logged)
    with torch.no_grad():
        module.validation_step(val_batch, batch_idx=0)

    assert "valid/loss" in first and "valid/x1_rmsd" in first
    for t in config.experiment.validation.t_values:
        assert f"valid/loss_t{t:g}" in first
    for key, value in first.items():
        assert torch.isfinite(value), key
        assert torch.equal(value, logged[key]), key

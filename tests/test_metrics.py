import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from astrafold.metrics import kabsch_rmsd, lddt, rna_tm_d0, tm_score


def _helix(length: int) -> np.ndarray:
    """An A-form-like helix of C1' atoms, about 5.9 Angstrom apart."""
    i = np.arange(length)
    angle = i * 2 * np.pi / 11
    return np.stack([9 * np.cos(angle), 9 * np.sin(angle), 2.8 * i], axis=-1)


def _move(x: np.ndarray, seed: int = 0) -> np.ndarray:
    rot = Rotation.random(random_state=seed).as_matrix()
    return x @ rot.T + np.array([10.0, -5.0, 3.0])


def test_identical_structures_score_perfectly():
    gt = _helix(40)
    moved = _move(gt)
    assert tm_score(moved, gt) == pytest.approx(1.0, abs=1e-6)
    assert lddt(moved, gt) == pytest.approx(1.0)
    rmsd = kabsch_rmsd(
        torch.tensor(moved)[None], torch.tensor(gt)[None], torch.ones(1, 40)
    )
    assert rmsd.item() == pytest.approx(0.0, abs=1e-3)


def test_tm_score_finds_the_well_predicted_part():
    """Half the structure right, half displaced: TM-score ~ 1/2, RMSD large."""
    gt = _helix(60)
    pred = gt.copy()
    pred[30:] += np.array([30.0, 0.0, 0.0])
    tm = tm_score(_move(pred, seed=1), gt)
    assert 0.45 < tm < 0.6
    # The best superposition is on one half; a plain Kabsch fit lies in between.
    rmsd = kabsch_rmsd(
        torch.tensor(pred)[None], torch.tensor(gt)[None], torch.ones(1, 60)
    )
    assert rmsd.item() > 5


def test_random_structures_score_low():
    rng = np.random.default_rng(0)
    gt = _helix(80)
    pred = np.cumsum(rng.normal(size=(80, 3)) * 3.4, axis=0)  # a random walk
    assert tm_score(pred, gt) < 0.3
    assert lddt(pred, gt) < 0.4


def test_lddt_ignores_global_displacement_of_distant_parts():
    gt = _helix(60)
    gt[30:] += np.array([100.0, 0.0, 0.0])  # two parts far apart
    pred = gt.copy()
    pred[30:] += np.array([50.0, 0.0, 0.0])  # moved, but each part is intact
    assert lddt(pred, gt) == pytest.approx(1.0)


def test_rna_d0():
    assert rna_tm_d0(10) == 0.3
    assert rna_tm_d0(100) == pytest.approx(0.6 * np.sqrt(99.5) - 2.5)

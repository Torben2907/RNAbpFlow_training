"""
Structure quality metrics between predicted and ground truth coordinates of one
representative atom per nucleotide (here the C1' frame origins), with known residue
correspondence (same sequence).

    kabsch_rmsd : RMSD after optimal superposition (global, sensitive to outliers)
    tm_score    : TM-score with the RNA d0 of US-align (global, length normalized)
    lddt        : lDDT of the representative atoms (local, superposition free)
"""

import math

import numpy as np
import torch
from torch import Tensor

from astrafold.data import utils as du


def kabsch_rmsd(pred: Tensor, gt: Tensor, mask: Tensor) -> Tensor:
    """Per-example RMSD [B] between [B, N, 3] coordinates over `mask` after optimal superposition."""
    mask = mask.float()
    num_res = mask.sum(dim=-1).clamp(min=1)  # [B]

    def center(x: Tensor) -> Tensor:
        centroid = (x * mask[..., None]).sum(dim=1) / num_res[:, None]
        return (x - centroid[:, None]) * mask[..., None]

    pred_c, gt_c = center(pred.float()), center(gt.float())
    pred_aligned, _, _ = du.batch_align_structures(pred_c, gt_c, mask=mask)
    sq_dist = ((pred_aligned - gt_c) ** 2).sum(dim=-1) * mask
    return torch.sqrt(sq_dist.sum(dim=-1) / num_res)


def rna_tm_d0(length: int) -> float:
    """The d0 that US-align uses for RNA (Gong et al. 2019), in Angstrom."""
    if length < 12:
        return 0.3
    if length < 16:
        return 0.4
    if length < 20:
        return 0.5
    if length < 24:
        return 0.6
    if length < 30:
        return 0.7
    return 0.6 * math.sqrt(length - 0.5) - 2.5


def _superpose(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotation R and translation t with mobile @ R.T + t ~ target (least squares)."""
    mu_m, mu_t = mobile.mean(axis=0), target.mean(axis=0)
    h = (mobile - mu_m).T @ (target - mu_t)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return rot, mu_t - mu_m @ rot.T


def tm_score(pred: np.ndarray, gt: np.ndarray, max_iterations: int = 20) -> float:
    """
    TM-score of `pred` [L, 3] against `gt` [L, 3] (normalized by L), maximized over
    superpositions as in the TM-score program: superpositions are seeded from
    fragments of length L, L/2, L/4, ... (at least 4) and refined iteratively on the
    residues closer than a search cutoff.
    """
    length = len(gt)
    d0 = rna_tm_d0(length)
    d0_search = min(max(d0, 4.5), 8.0)

    def score_of(rot: np.ndarray, trans: np.ndarray) -> tuple[float, np.ndarray]:
        dist = np.linalg.norm(pred @ rot.T + trans - gt, axis=-1)
        return float(np.mean(1.0 / (1.0 + (dist / d0) ** 2))), dist

    best = 0.0
    fragment = length
    while True:
        fragment = max(fragment, min(4, length))
        step = max(1, fragment // 2)
        for start in range(0, length - fragment + 1, step):
            selected = np.zeros(length, dtype=bool)
            selected[start : start + fragment] = True
            for _ in range(max_iterations):
                rot, trans = _superpose(pred[selected], gt[selected])
                score, dist = score_of(rot, trans)
                best = max(best, score)
                cutoff = d0_search
                new_selected = dist < cutoff
                while new_selected.sum() < 3 and cutoff < 100:
                    cutoff += 0.5
                    new_selected = dist < cutoff
                if np.array_equal(new_selected, selected):
                    break
                selected = new_selected
        if fragment <= 4:
            break
        fragment //= 2
    return best


def lddt(
    pred: np.ndarray,
    gt: np.ndarray,
    inclusion_radius: float = 15.0,
    thresholds: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
) -> float:
    """
    lDDT of `pred` [L, 3] against `gt` [L, 3]: the share of ground truth distances
    below `inclusion_radius` (between different residues) that `pred` preserves
    within each threshold, averaged over thresholds and residues.
    """
    d_gt = np.linalg.norm(gt[:, None] - gt[None], axis=-1)
    d_pred = np.linalg.norm(pred[:, None] - pred[None], axis=-1)
    pairs = (d_gt < inclusion_radius) & ~np.eye(len(gt), dtype=bool)
    error = np.abs(d_gt - d_pred)
    preserved = np.mean([(error < t) for t in thresholds], axis=0)  # [L, L]
    has_pairs = pairs.any(axis=-1)
    per_residue = (preserved * pairs).sum(-1)[has_pairs] / pairs.sum(-1)[has_pairs]
    return float(per_residue.mean()) if per_residue.size else float("nan")

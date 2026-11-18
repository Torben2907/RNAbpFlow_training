"""
Samples all-atom RNA structures for sequences with a trained checkpoint.

Adapted from the sampler of the original RNAbpFlow inference code
(https://github.com/Bhattacharya-Lab/RNAbpFlow).
"""

import csv
import re
from pathlib import Path

import numpy as np
import torch

import astrafold.utils as eu
from astrafold.config.hydra_schema import HybridSamplerConfig, InferenceConfig
from astrafold.data.all_atom import to_atom28_rna
from astrafold.data.hybrid_sampler import sample_hybrid
from astrafold.data.nucleic_acid import OUTPUT_FORMATS, write_structure
from astrafold.data.sequence import read_fasta, sequence_features
from astrafold.models.flow_module import FlowModule

log = eu.get_pylogger(__name__)

SAMPLERS = ("flow", "hybrid")


def resolve_device(device: str) -> torch.device:
    """ "auto" picks the GPU if there is one; anything else is passed to torch."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class Sampler:
    """
    Loads the model of a training checkpoint once and samples structures with it.

    Params:
        ckpt_path     : checkpoint written by train.py
        device        : "auto", "cpu", "cuda" or "cuda:<id>"
        num_timesteps : sampler steps; None uses the value stored in the checkpoint
        sampler       : "flow" (one trajectory) or "hybrid" (see hybrid_sampler.py)
        hybrid        : options of the hybrid sampler; None uses the defaults
    """

    def __init__(
        self,
        ckpt_path: str | Path,
        device: str = "auto",
        num_timesteps: int | None = None,
        sampler: str = "flow",
        hybrid: HybridSamplerConfig | None = None,
    ):
        if sampler not in SAMPLERS:
            raise ValueError(f"Unknown sampler {sampler!r}, choose from {SAMPLERS}")
        self.device = resolve_device(device)
        self._num_timesteps = num_timesteps
        self._sampler = sampler
        self._hybrid = hybrid if hybrid is not None else HybridSamplerConfig()
        self._module = FlowModule.load_from_checkpoint(
            ckpt_path, map_location=self.device
        )
        self._module.to(self.device).eval()  # eval mode turns dropout off
        log.info(f"Loaded {ckpt_path} on {self.device}")

    def sample(self, sequence: str, seeds: list[int]) -> dict[str, np.ndarray]:
        """
        Generates one structure per seed for an RNA sequence of length N.

        Returns:
            atom_positions : [len(seeds), N, 28, 3] coordinates in Angstrom, in the
                             atom order of base_constants.atom_types
            atom_mask      : [N, 28] 1 where the residue has that atom
            aatype         : [N] residue type index into base_constants.restypes
            contact_tokens : [len(seeds), N] generated contact tokens (1 = contact)

        The hybrid sampler adds `self_consistency`, `reveal_t`, `reveal_round`,
        `num_forced` and `num_model_calls` (see `sample_hybrid`).
        """
        feats = {k: v.to(self.device) for k, v in sequence_features(sequence).items()}
        if self._sampler == "hybrid":
            frames = sample_hybrid(
                self._module.interpolant,
                self._module.model,
                feats["onehot"],
                seeds=seeds,
                num_timesteps=self._num_timesteps,
                confidence_tolerance=self._hybrid.confidence_tolerance,
                tokens_per_round=self._hybrid.tokens_per_round,
                remask_fraction=self._hybrid.remask_fraction,
            )
        else:
            frames = self._module.interpolant.sample(
                self._module.model,
                feats["onehot"],
                seeds=seeds,
                num_timesteps=self._num_timesteps,
            )

        num_batch = len(seeds)
        atom_positions = to_atom28_rna(
            frames["trans"],
            frames["rotmats"],
            feats["aatype"].expand(num_batch, -1),
            frames["torsions"],
            feats["atom28_to_atom23"].expand(num_batch, -1, -1),
            feats["atom28_mask"].expand(num_batch, -1, -1),
        )
        result = {
            "atom_positions": atom_positions.cpu().numpy(),
            "atom_mask": feats["atom28_mask"].cpu().numpy(),
            "aatype": feats["aatype"].cpu().numpy(),
        }
        for key in _SAMPLE_OUTPUTS:
            if key in frames:
                result[key] = frames[key].cpu().numpy()
        return result


# Per-sample outputs of the samplers that `Sampler.sample` passes on.
_SAMPLE_OUTPUTS = (
    "contact_tokens",
    "self_consistency",
    "reveal_t",
    "reveal_round",
    "num_forced",
    "num_model_calls",
)
_SCORE_COLUMNS = ("sample", "self_consistency", "num_forced", "num_model_calls")


def _write_scores(path: Path, rows: list[dict]) -> None:
    """Writes the per-sample scores of the hybrid sampler, best sample first."""
    rows = sorted(rows, key=lambda row: row["self_consistency"], reverse=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_SCORE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _directory_name(record_id: str) -> str:
    """Makes a FASTA record id safe to use as a directory name."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", record_id)


def run_inference(cfg: InferenceConfig) -> list[Path]:
    """
    Samples `cfg.num_samples` structures for every sequence in `cfg.fasta_path` and
    writes them to `<cfg.output_dir>/<record id>/Sample_<i>.<ext>`. The hybrid
    sampler also writes `scores.csv` there, with the samples ranked by their
    self-consistency score. Returns the written paths.
    """
    if cfg.output_format not in OUTPUT_FORMATS:
        raise ValueError(
            f"Unknown output_format {cfg.output_format!r}, choose from {OUTPUT_FORMATS}"
        )
    if cfg.sampler not in SAMPLERS:
        raise ValueError(f"Unknown sampler {cfg.sampler!r}, choose from {SAMPLERS}")
    records = read_fasta(cfg.fasta_path)
    sampler = Sampler(
        cfg.ckpt_path, cfg.device, cfg.num_timesteps, cfg.sampler, cfg.hybrid
    )

    written: list[Path] = []
    for record_id, sequence in records:
        target_dir = Path(cfg.output_dir) / _directory_name(record_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"Sampling {cfg.num_samples} structures for {record_id}")

        scores: list[dict] = []
        for start in range(0, cfg.num_samples, cfg.batch_size):
            sample_ids = range(start, min(start + cfg.batch_size, cfg.num_samples))
            result = sampler.sample(sequence, [cfg.seed + i for i in sample_ids])
            for sample_id, atom_positions in zip(
                sample_ids, result["atom_positions"], strict=True
            ):
                written += write_structure(
                    target_dir / f"Sample_{sample_id}",
                    result["aatype"],
                    atom_positions,
                    result["atom_mask"],
                    cfg.output_format,
                )
            if "self_consistency" in result:
                for j, sample_id in enumerate(sample_ids):
                    scores.append(
                        {"sample": f"Sample_{sample_id}"}
                        | {k: result[k][j].item() for k in _SCORE_COLUMNS[1:]}
                    )

        if scores:
            _write_scores(target_dir / "scores.csv", scores)
            written.append(target_dir / "scores.csv")

    log.info(f"Wrote {len(written)} files to {cfg.output_dir}")
    return written

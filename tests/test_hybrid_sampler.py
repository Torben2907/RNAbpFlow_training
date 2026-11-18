import csv
import math

import numpy as np
import pytest
import torch

from astrafold.config.hydra_schema import (
    HybridSamplerConfig,
    HydraConfig,
    InferenceConfig,
)
from astrafold.data.contacts import CONTACT, MASK, NO_CONTACT
from astrafold.data.hybrid_sampler import sample_hybrid
from astrafold.data.sequence import sequence_features
from astrafold.inference import Sampler, run_inference
from astrafold.models.flow_module import FlowModule

SEQUENCE = "GGACUUCGGUCC"
N = len(SEQUENCE)
NUM_TIMESTEPS = 4


@pytest.fixture(scope="module")
def flow_module(config: HydraConfig) -> FlowModule:
    torch.manual_seed(0)
    return FlowModule(config).eval()


def _onehot(sequence: str = SEQUENCE) -> torch.Tensor:
    return sequence_features(sequence)["onehot"]


def _hybrid(flow_module, model=None, seeds=(0,), **kwargs):
    kwargs.setdefault("num_timesteps", NUM_TIMESTEPS)
    return sample_hybrid(
        flow_module.interpolant,
        model if model is not None else flow_module.model,
        _onehot(),
        seeds=list(seeds),
        **kwargs,
    )


class Oracle:
    """
    A stand-in model with fixed per-position token probabilities. Predicts
    p(CONTACT) = p_contact[i] for every nucleotide i (and x1 = 0), and records the
    inputs of every call.
    """

    def __init__(self, p_contact: torch.Tensor):
        self.logits = torch.stack([torch.log1p(-p_contact), torch.log(p_contact)], -1)
        self.calls: list[dict[str, torch.Tensor]] = []

    def __call__(self, batch):
        self.calls.append({k: v.clone() for k, v in batch.items()})
        b = batch["trans_t"].shape[0]
        return {
            "pred_trans": torch.zeros(b, N, 3),
            "pred_rotmats": torch.eye(3).expand(b, N, 3, 3),
            "pred_torsions": torch.zeros(b, N, 18),
            "token_logits": self.logits.expand(b, -1, -1),
        }


def test_hybrid_shapes_and_tokens(flow_module):
    out = _hybrid(flow_module, seeds=[0, 1, 2])
    assert out["trans"].shape == (3, N, 3)
    assert out["rotmats"].shape == (3, N, 3, 3)
    assert out["torsions"].shape == (3, N, 9, 2)
    for key in ("contact_tokens", "remask", "reveal_t", "reveal_round"):
        assert out[key].shape == (3, N), key
    for key in ("self_consistency", "num_forced", "num_model_calls"):
        assert out[key].shape == (3,), key
    for key in ("trans", "rotmats", "torsions", "self_consistency", "reveal_t"):
        assert torch.isfinite(out[key]).all(), key

    tokens = out["contact_tokens"]
    assert torch.all((tokens == CONTACT) | (tokens == NO_CONTACT))
    assert torch.all(out["self_consistency"] <= 0)
    # One token per round: the rounds are a permutation of 0..N-1.
    assert torch.equal(
        out["reveal_round"].sort(dim=-1).values, torch.arange(N).expand(3, N)
    )


def test_confident_tokens_are_revealed_at_the_start_in_confidence_order(flow_module):
    # All tokens confident from the start: one model call per token, no Euler step.
    p_contact = torch.linspace(0.95, 0.999, N)
    p_contact[::2] = 1 - p_contact[::2]  # half of them confident NO_CONTACT
    oracle = Oracle(p_contact)
    out = _hybrid(flow_module, oracle, confidence_tolerance=0.1)

    min_t = flow_module.interpolant._cfg.min_t
    assert torch.allclose(out["reveal_t"], torch.full((1, N), min_t))
    assert out["num_forced"].item() == 0
    assert out["num_model_calls"].item() == N + NUM_TIMESTEPS + 1

    confidence = torch.maximum(p_contact, 1 - p_contact)
    expected_order = confidence.argsort(descending=True)
    assert torch.equal(out["reveal_round"][0].argsort(), expected_order)
    expected_tokens = torch.where(p_contact > 0.5, CONTACT, NO_CONTACT)
    assert torch.equal(out["contact_tokens"][0], expected_tokens)

    # Every round sees exactly one more revealed token than the round before.
    for i, call in enumerate(oracle.calls[:N]):
        assert (call["contact_tokens"] != MASK).sum().item() == i


def test_unconfident_tokens_run_to_the_end_and_are_forced(flow_module):
    oracle = Oracle(torch.full((N,), 0.6))  # never reaches 1 - eps = 0.9
    out = _hybrid(flow_module, oracle, confidence_tolerance=0.1)

    assert out["num_forced"].item() == N
    assert torch.allclose(out["reveal_t"], torch.ones(1, N))
    # N full trajectories, the final trajectory and the scoring call.
    assert out["num_model_calls"].item() == (N + 1) * NUM_TIMESTEPS + 1
    assert torch.all(out["contact_tokens"] == CONTACT)


def test_every_trajectory_restarts_from_the_same_noise(flow_module):
    oracle = Oracle(torch.full((N,), 0.6))
    _hybrid(flow_module, oracle)
    min_t = flow_module.interpolant._cfg.min_t

    starts = [c for c in oracle.calls if c["t"][0, 0].item() == pytest.approx(min_t)]
    assert len(starts) == N + 1  # N reveal trajectories and the final one
    for call in starts[1:]:
        assert torch.equal(call["trans_t"], starts[0]["trans_t"])
        assert torch.equal(call["rotmats_t"], starts[0]["rotmats_t"])
    # Within a trajectory the flow time increases step by step.
    ts = [c["t"][0, 0].item() for c in oracle.calls[:NUM_TIMESTEPS]]
    assert ts == pytest.approx(np.linspace(min_t, 1, NUM_TIMESTEPS).tolist())


def test_tokens_per_round_reveals_several_confident_tokens(flow_module):
    p_contact = torch.full((N,), 0.6)
    p_contact[:5] = 0.99  # five confident tokens, the rest not
    out = _hybrid(flow_module, Oracle(p_contact), tokens_per_round=3)

    # Rounds: 3 confident, 2 confident, then the 7 others one by one (forced).
    assert out["reveal_round"][0, :5].sort().values.tolist() == [0, 0, 0, 1, 1]
    assert out["reveal_round"][0, 5:].sort().values.tolist() == list(range(2, 9))
    assert out["num_forced"].item() == N - 5


def test_self_consistency_of_a_confident_model(flow_module):
    p_contact = torch.linspace(0.6, 0.99, N)
    out = _hybrid(flow_module, Oracle(p_contact), remask_fraction=1.0)
    # All tokens are CONTACT (p > 0.5), so the score is the sum of log p_contact.
    assert out["self_consistency"].item() == pytest.approx(
        torch.log(p_contact).sum().item(), rel=1e-5
    )

    out = _hybrid(flow_module, Oracle(p_contact), remask_fraction=0.25)
    remask = out["remask"][0]
    assert remask.sum().item() == math.ceil(0.25 * N)
    assert out["self_consistency"].item() == pytest.approx(
        torch.log(p_contact[remask]).sum().item(), rel=1e-5
    )


def test_scoring_call_sees_the_final_structure_and_remasked_tokens(flow_module):
    oracle = Oracle(torch.full((N,), 0.99))
    out = _hybrid(flow_module, oracle, remask_fraction=0.5)
    score_call = oracle.calls[-1]
    assert torch.all(score_call["t"] == 1.0)
    assert torch.equal(score_call["trans_t"], out["trans"])
    assert torch.equal(score_call["contact_tokens"] == MASK, out["remask"])


def test_hybrid_is_deterministic_and_independent_of_the_batch(flow_module):
    first = _hybrid(flow_module, seeds=[3, 4, 5])
    second = _hybrid(flow_module, seeds=[3, 4, 5])
    for key in first:
        assert torch.equal(first[key], second[key]), key

    alone = _hybrid(flow_module, seeds=[4])
    assert torch.allclose(first["trans"][1], alone["trans"][0], atol=1e-2)
    assert torch.equal(first["contact_tokens"][1], alone["contact_tokens"][0])
    assert torch.equal(first["reveal_round"][1], alone["reveal_round"][0])


def test_hybrid_does_not_touch_the_global_rng(flow_module):
    torch.manual_seed(1234)
    rng_before = torch.get_rng_state()
    _hybrid(flow_module, seeds=[0])
    assert torch.equal(torch.get_rng_state(), rng_before)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_timesteps": 1}, "at least 2 timesteps"),
        ({"confidence_tolerance": 1.0}, "confidence_tolerance"),
        ({"tokens_per_round": 0}, "tokens_per_round"),
        ({"remask_fraction": 0.0}, "remask_fraction"),
    ],
)
def test_hybrid_rejects_invalid_options(flow_module, kwargs, message):
    with pytest.raises(ValueError, match=message):
        _hybrid(flow_module, **kwargs)


def test_run_inference_with_the_hybrid_sampler(flow_module, tmp_path):
    from pytorch_lightning import Trainer

    ckpt = tmp_path / "model.ckpt"
    trainer = Trainer(accelerator="cpu", logger=False, enable_checkpointing=False)
    trainer.strategy.connect(flow_module)
    trainer.save_checkpoint(ckpt)

    fasta = tmp_path / "targets.fasta"
    fasta.write_text(f">hairpin\n{SEQUENCE}\n")
    cfg = InferenceConfig(
        ckpt_path=str(ckpt),
        fasta_path=str(fasta),
        output_dir=str(tmp_path / "predictions"),
        num_samples=3,
        batch_size=2,
        num_timesteps=NUM_TIMESTEPS,
        seed=0,
        output_format="pdb",
        device="cpu",
        sampler="hybrid",
        hybrid=HybridSamplerConfig(confidence_tolerance=0.2),
    )
    written = run_inference(cfg)
    target = tmp_path / "predictions" / "hairpin"
    assert sorted(written) == sorted(
        [target / f"Sample_{i}.pdb" for i in range(3)] + [target / "scores.csv"]
    )

    with open(target / "scores.csv") as f:
        rows = list(csv.DictReader(f))
    assert sorted(row["sample"] for row in rows) == [f"Sample_{i}" for i in range(3)]
    scores = [float(row["self_consistency"]) for row in rows]
    assert scores == sorted(scores, reverse=True)  # best sample first

    # The scores match a direct call with the same seeds.
    sampler = Sampler(ckpt, "cpu", NUM_TIMESTEPS, "hybrid", HybridSamplerConfig(0.2))
    direct = sampler.sample(SEQUENCE, seeds=[0, 1, 2])["self_consistency"]
    by_name = {row["sample"]: float(row["self_consistency"]) for row in rows}
    for i in range(3):
        assert by_name[f"Sample_{i}"] == pytest.approx(direct[i], abs=1e-4)


def test_unknown_sampler_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown sampler"):
        Sampler("does/not/exist.ckpt", "cpu", sampler="discrete")

import copy
import itertools
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from scipy.spatial.transform import Rotation

from astrafold.config.hydra_schema import HydraConfig, InferenceConfig
from astrafold.data import base_constants as rc
from astrafold.data import rigid_utils
from astrafold.data import utils as du
from astrafold.data.all_atom import to_atom28_rna
from astrafold.data.contacts import CONTACT, MASK, NO_CONTACT, unmask_tokens
from astrafold.data.nucleic_acid import to_pdb, write_structure
from astrafold.data.sequence import (
    normalize_sequence,
    read_fasta,
    sequence_features,
)
from astrafold.inference import Sampler, run_inference
from astrafold.models.flow_module import FlowModule

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEQUENCE = "GGACUUCGGUCC"
NUM_TIMESTEPS = 4


# --- Sequence features -----------------------------------------------------------


def test_normalize_sequence():
    assert normalize_sequence(" acgu\n") == "ACGU"
    assert normalize_sequence("ACGT") == "ACGU"  # T is read as U
    with pytest.raises(ValueError, match="N"):
        normalize_sequence("ACGNU")
    with pytest.raises(ValueError, match="Empty"):
        normalize_sequence("")


def test_sequence_features_shapes():
    feats = sequence_features(SEQUENCE)
    n = len(SEQUENCE)
    assert feats["onehot"].shape == (n, 4) and feats["onehot"].dtype == torch.float32
    assert torch.all(feats["onehot"].sum(dim=-1) == 1)
    assert feats["aatype"].shape == (n,) and feats["aatype"].dtype == torch.long
    assert feats["atom28_to_atom23"].shape == (n, 28)
    assert feats["atom28_mask"].shape == (n, 28)
    # A, G, C and U have 22, 23, 20 and 20 heavy atoms.
    num_atoms = {"A": 22, "G": 23, "C": 20, "U": 20}
    assert feats["atom28_mask"].sum(dim=-1).tolist() == [
        num_atoms[nt] for nt in SEQUENCE
    ]


@pytest.fixture(scope="module")
def training_examples() -> list[tuple[dict, dict]]:
    """The two feature pickles of the first few training chains."""
    csv = pd.read_csv(PROJECT_ROOT / "metadata" / "RNA3DB_train.csv").head(5)
    examples = []
    for name, processed_path in zip(csv.pdb_name, csv.processed_path, strict=True):
        feats = du.read_pkl(str(PROJECT_ROOT / processed_path))
        nufeat = du.read_pkl(f"{PROJECT_ROOT}/pkls/{name}_base/{name}_feat.pkl")
        examples.append((feats, nufeat))
    return examples


def test_sequence_features_match_training_data(training_examples):
    """Inference must encode a sequence exactly like the training pickles do."""
    for feats, nufeat in training_examples:
        sequence = "".join(rc.restypes[i] for i in nufeat["aatype"])
        seq_feats = sequence_features(sequence)

        assert torch.equal(seq_feats["aatype"], nufeat["aatype"])
        assert np.array_equal(seq_feats["onehot"].numpy(), feats["onehot"])
        assert torch.equal(
            seq_feats["atom28_to_atom23"], nufeat["residx_atom28_to_atom23"]
        )
        assert torch.equal(seq_feats["atom28_mask"], nufeat["atom28_atom_exists"])


def test_read_fasta(tmp_path):
    fasta = tmp_path / "in.fasta"
    fasta.write_text(">first some description\nGGAC\nUUCG\n>second\nACGU\n")
    assert read_fasta(fasta) == [("first", "GGACUUCG"), ("second", "ACGU")]

    empty = tmp_path / "empty.fasta"
    empty.write_text("")
    with pytest.raises(ValueError, match="No FASTA records"):
        read_fasta(empty)


# --- All-atom construction and output files --------------------------------------


def test_all_atom_reconstruction_matches_ground_truth(training_examples):
    """Ground truth frames and torsions must rebuild the ground truth atoms."""
    for _, nufeat in training_examples:
        seq_feats = sequence_features("".join(rc.restypes[i] for i in nufeat["aatype"]))
        frames = rigid_utils.Rigid.from_tensor_4x4(nufeat["rigidgroups_gt_frames"])[
            :, 0
        ]
        atom_positions = to_atom28_rna(
            frames.get_trans(),
            frames.get_rots().get_rot_mats(),
            nufeat["aatype"],
            nufeat["torsion_angles_sin_cos"][:, :9],
            seq_feats["atom28_to_atom23"],
            seq_feats["atom28_mask"],
        )
        assert atom_positions.shape == (len(nufeat["aatype"]), 28, 3)

        # The frames are centered, the stored atoms are not: remove the offset, taken
        # from C1', which is the origin of the backbone frame.
        mask = nufeat["all_atom_mask"].bool() & seq_feats["atom28_mask"].bool()
        diff = atom_positions - nufeat["all_atom_positions"]
        c1 = rc.atom_order["C1'"]
        diff = diff - diff[:, c1][mask[:, c1]].mean(dim=0)
        dist = diff.norm(dim=-1)

        # The atoms of the backbone frame itself must be (almost) exact.
        for atom in ("C1'", "O4'", "C2'"):
            idx = rc.atom_order[atom]
            assert dist[:, idx][mask[:, idx]].mean() < 0.05, atom
        # Ideal bond geometry puts the other atoms within a few tenths of an Angstrom.
        assert torch.sqrt((dist[mask] ** 2).mean()) < 0.6
        # Atoms a residue does not have are zero.
        assert torch.all(atom_positions[~seq_feats["atom28_mask"].bool()] == 0)


def _random_structure(sequence: str = SEQUENCE):
    feats = sequence_features(sequence)
    rng = np.random.default_rng(0)
    atom_positions = rng.normal(scale=20.0, size=(len(sequence), 28, 3))
    return feats["aatype"].numpy(), atom_positions, feats["atom28_mask"].numpy()


def test_to_pdb_round_trip():
    aatype, atom_positions, atom_mask = _random_structure()
    structure = gemmi.read_pdb_string(to_pdb(aatype, atom_positions, atom_mask))

    assert len(structure) == 1 and len(structure[0]) == 1
    chain = structure[0][0]
    assert chain.name == "A"
    assert [residue.name for residue in chain] == list(SEQUENCE)
    assert [residue.seqid.num for residue in chain] == list(range(1, len(SEQUENCE) + 1))

    for i, residue in enumerate(chain):
        expected_names = [n for n, m in zip(rc.atom_types, atom_mask[i]) if m]
        assert [atom.name for atom in residue] == expected_names
        for atom in residue:
            expected_pos = atom_positions[i, rc.atom_order[atom.name]]
            assert np.allclose(atom.pos.tolist(), expected_pos, atol=1e-3)
            assert atom.element.name == atom.name[0]


def test_to_pdb_rejects_invalid_residue_types():
    aatype, atom_positions, atom_mask = _random_structure()
    aatype = aatype.copy()
    aatype[0] = 4
    with pytest.raises(ValueError, match="Invalid residue types"):
        to_pdb(aatype, atom_positions, atom_mask)


@pytest.mark.parametrize(
    ("output_format", "suffixes"),
    [("pdb", [".pdb"]), ("mmcif", [".cif"]), ("both", [".pdb", ".cif"])],
)
def test_write_structure_formats(tmp_path, output_format, suffixes):
    aatype, atom_positions, atom_mask = _random_structure()
    written = write_structure(
        tmp_path / "Sample_0", aatype, atom_positions, atom_mask, output_format
    )
    assert [path.name for path in written] == [f"Sample_0{s}" for s in suffixes]

    num_atoms = int(atom_mask.sum())
    for path in written:
        structure = gemmi.read_structure(str(path))
        assert structure[0].count_atom_sites() == num_atoms
        assert [residue.name for residue in structure[0][0]] == list(SEQUENCE)


def test_write_structure_rejects_unknown_format(tmp_path):
    with pytest.raises(ValueError, match="Unknown output format"):
        write_structure(tmp_path / "x", *_random_structure(), output_format="xyz")
    assert not list(tmp_path.iterdir())


# --- Token unmasking -------------------------------------------------------------


def _confident_logits(values: torch.Tensor) -> torch.Tensor:
    """Logits that put (almost) all probability on `values`."""
    return torch.nn.functional.one_hot(values, num_classes=2).float() * 100


def test_unmask_tokens_reveal_probability_extremes():
    tokens = torch.tensor([[MASK, CONTACT, MASK, NO_CONTACT, MASK]])
    predicted = torch.tensor([[1, 0, 0, 1, 1]])
    logits = _confident_logits(predicted)

    assert torch.equal(unmask_tokens(tokens, logits, reveal_probability=0.0), tokens)

    revealed = unmask_tokens(tokens, logits, reveal_probability=1.0)
    # Masked tokens take the predicted value, revealed ones keep theirs.
    assert revealed.tolist() == [[1, CONTACT, 0, NO_CONTACT, 1]]


def test_unmask_tokens_uses_the_given_noise():
    tokens = torch.full((1, 4), MASK)
    logits = torch.zeros(1, 4, 2)  # p(no contact) = p(contact) = 0.5
    revealed = unmask_tokens(
        tokens,
        logits,
        reveal_probability=0.5,
        reveal_noise=torch.tensor([[0.1, 0.9, 0.4, 0.6]]),
        value_noise=torch.tensor([[0.2, 0.2, 0.8, 0.8]]),
    )
    assert revealed.tolist() == [[NO_CONTACT, MASK, CONTACT, MASK]]


def test_unmask_tokens_reveals_the_expected_fraction():
    torch.manual_seed(0)
    tokens = torch.full((1, 20_000), MASK)
    logits = torch.tensor([0.0, np.log(3.0)]).expand(1, 20_000, 2)  # p(contact) = 0.75
    revealed = unmask_tokens(tokens, logits, reveal_probability=0.3)

    is_revealed = revealed != MASK
    assert abs(is_revealed.float().mean() - 0.3) < 0.02
    assert abs((revealed[is_revealed] == CONTACT).float().mean() - 0.75) < 0.03


# --- Sampling --------------------------------------------------------------------


@pytest.fixture(scope="module")
def flow_module(config: HydraConfig) -> FlowModule:
    torch.manual_seed(0)
    return FlowModule(config).eval()


def _sample(flow_module: FlowModule, seeds: list[int], sequence: str = SEQUENCE):
    onehot = sequence_features(sequence)["onehot"]
    return flow_module.interpolant.sample(
        flow_module.model, onehot, seeds=seeds, num_timesteps=NUM_TIMESTEPS
    )


def test_sample_shapes(flow_module):
    sample = _sample(flow_module, seeds=[0, 1, 2])
    n = len(SEQUENCE)
    assert sample["trans"].shape == (3, n, 3)
    assert sample["rotmats"].shape == (3, n, 3, 3)
    assert sample["torsions"].shape == (3, n, 9, 2)
    assert sample["contact_tokens"].shape == (3, n)
    for key, value in sample.items():
        assert torch.isfinite(value).all(), key

    # Valid rotations, and every token is revealed at the end.
    identity = torch.eye(3).expand(3, n, 3, 3)
    rotmats = sample["rotmats"]
    assert torch.allclose(rotmats @ rotmats.transpose(-1, -2), identity, atol=1e-4)
    assert torch.all(
        (sample["contact_tokens"] == CONTACT) | (sample["contact_tokens"] == NO_CONTACT)
    )


def test_sample_goes_from_masked_to_revealed_tokens(flow_module):
    seen_tokens: list[torch.Tensor] = []
    seen_t: list[float] = []

    def record(module, args):
        seen_tokens.append(args[0]["contact_tokens"].clone())
        seen_t.append(args[0]["t"][0, 0].item())

    hook = flow_module.model.register_forward_pre_hook(record)
    try:
        sample = _sample(flow_module, seeds=[0, 1], sequence=SEQUENCE * 4)
    finally:
        hook.remove()

    # One model call per flow time, from min_t to 1.
    assert len(seen_tokens) == NUM_TIMESTEPS
    assert seen_t == sorted(seen_t)
    assert seen_t[0] == pytest.approx(flow_module.interpolant._cfg.min_t)
    assert seen_t[-1] == pytest.approx(1.0)

    assert torch.all(seen_tokens[0] == MASK)
    assert not torch.any(seen_tokens[-1] == MASK)
    assert torch.equal(seen_tokens[-1], sample["contact_tokens"])
    for before, after in itertools.pairwise(seen_tokens):
        # A revealed token never changes again.
        was_revealed = before != MASK
        assert torch.equal(after[was_revealed], before[was_revealed])


def test_sample_is_deterministic_per_seed(flow_module):
    first = _sample(flow_module, seeds=[7, 8])
    second = _sample(flow_module, seeds=[7, 8])
    for key in first:
        assert torch.equal(first[key], second[key]), key

    # Different seeds give different structures.
    assert not torch.allclose(first["trans"][0], first["trans"][1], atol=1e-2)


def test_sample_does_not_depend_on_batch_composition(flow_module):
    batched = _sample(flow_module, seeds=[3, 4, 5])
    alone = _sample(flow_module, seeds=[4])
    assert torch.allclose(batched["trans"][1], alone["trans"][0], atol=1e-2)
    assert torch.allclose(batched["rotmats"][1], alone["rotmats"][0], atol=1e-3)
    assert torch.equal(batched["contact_tokens"][1], alone["contact_tokens"][0])


def test_sample_does_not_touch_the_global_rng(flow_module):
    torch.manual_seed(1234)
    rng_before = torch.get_rng_state()
    _sample(flow_module, seeds=[0])
    assert torch.equal(torch.get_rng_state(), rng_before)


def test_sample_feeds_back_self_conditioning(config: HydraConfig):
    cfg = copy.deepcopy(config)
    cfg.interpolant.self_condition = True
    module = FlowModule(cfg).eval()

    seen_sc: list[torch.Tensor | None] = []
    outputs: list[torch.Tensor] = []
    module.model.register_forward_pre_hook(
        lambda m, args: seen_sc.append(args[0].get("trans_sc"))
    )
    module.model.register_forward_hook(
        lambda m, args, output: outputs.append(output["pred_trans"])
    )
    _sample(module, seeds=[0])

    # The first call has no previous prediction; later calls get the last one.
    assert seen_sc[0] is None
    for sc, previous in zip(seen_sc[1:], outputs[:-1], strict=True):
        assert torch.equal(sc, previous)


def test_sample_converges_to_a_perfect_prediction(flow_module):
    """With a model that always predicts the same x1, the flow must arrive there."""
    n = len(SEQUENCE)
    torch.manual_seed(0)
    x1_trans = torch.randn(1, n, 3) * 10
    x1_rotmats = torch.tensor(Rotation.random(n).as_matrix(), dtype=torch.float32)[None]
    final_input: dict[str, torch.Tensor] = {}

    def oracle(batch):
        final_input.update(trans_t=batch["trans_t"], rotmats_t=batch["rotmats_t"])
        return {
            "pred_trans": x1_trans,
            "pred_rotmats": x1_rotmats,
            "pred_torsions": torch.zeros(1, n, 9, 2),
            "token_logits": torch.zeros(1, n, 2),
        }

    onehot = sequence_features(SEQUENCE)["onehot"]
    flow_module.interpolant.sample(oracle, onehot, seeds=[0], num_timesteps=50)

    # The frames given to the last model call (t = 1) are the integrated flow.
    assert torch.allclose(final_input["trans_t"], x1_trans, atol=1e-3)
    assert torch.allclose(final_input["rotmats_t"], x1_rotmats, atol=1e-3)


def test_sample_needs_two_timesteps(flow_module):
    onehot = sequence_features(SEQUENCE)["onehot"]
    with pytest.raises(ValueError, match="at least 2 timesteps"):
        flow_module.interpolant.sample(
            flow_module.model, onehot, seeds=[0], num_timesteps=1
        )


# --- End to end ------------------------------------------------------------------


@pytest.fixture(scope="module")
def checkpoint(flow_module, tmp_path_factory) -> Path:
    """A checkpoint as train.py writes it (untrained weights)."""
    path = tmp_path_factory.mktemp("ckpt") / "model.ckpt"
    trainer = Trainer(
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.strategy.connect(flow_module)
    trainer.save_checkpoint(path)
    return path


def _bond_lengths(structure: gemmi.Structure, atom_1: str, atom_2: str) -> list[float]:
    return [
        residue[atom_1][0].pos.dist(residue[atom_2][0].pos)
        for residue in structure[0][0]
    ]


def test_sampler_loads_checkpoint_in_eval_mode(checkpoint, flow_module):
    sampler = Sampler(checkpoint, device="cpu", num_timesteps=NUM_TIMESTEPS)
    assert not sampler._module.training
    for (name, loaded), (_, original) in zip(
        sampler._module.state_dict().items(),
        flow_module.state_dict().items(),
        strict=True,
    ):
        assert torch.equal(loaded, original), name

    result = sampler.sample(SEQUENCE, seeds=[0, 1])
    n = len(SEQUENCE)
    assert result["atom_positions"].shape == (2, n, 28, 3)
    assert result["atom_mask"].shape == (n, 28)
    assert result["aatype"].shape == (n,)
    assert result["contact_tokens"].shape == (2, n)
    assert np.isfinite(result["atom_positions"]).all()


def test_run_inference_end_to_end(checkpoint, tmp_path):
    fasta = tmp_path / "targets.fasta"
    fasta.write_text(f">rna/1 test\n{SEQUENCE}\n>rna2\nacgt\n")
    cfg = InferenceConfig(
        ckpt_path=str(checkpoint),
        fasta_path=str(fasta),
        output_dir=str(tmp_path / "predictions"),
        num_samples=3,
        batch_size=2,  # forces a full and a partial batch
        num_timesteps=NUM_TIMESTEPS,
        seed=42,
        output_format="both",
        device="cpu",
    )
    written = run_inference(cfg)

    out = tmp_path / "predictions"
    expected = [
        out / target / f"Sample_{i}{suffix}"
        for target in ("rna_1", "rna2")  # the "/" of the record id is replaced
        for i in range(3)
        for suffix in (".pdb", ".cif")
    ]
    assert sorted(written) == sorted(expected)
    assert all(path.exists() for path in expected)

    for target, sequence in (("rna_1", SEQUENCE), ("rna2", "ACGU")):
        structures = [
            gemmi.read_structure(str(out / target / f"Sample_{i}.pdb"))
            for i in range(3)
        ]
        for structure in structures:
            assert [residue.name for residue in structure[0][0]] == list(sequence)
            # Atoms of the same rigid group keep their ideal distances, whatever the
            # (untrained) model predicts.
            assert np.allclose(_bond_lengths(structure, "C1'", "C2'"), 1.53, atol=0.03)
            assert np.allclose(_bond_lengths(structure, "C1'", "O4'"), 1.42, atol=0.03)

        # The samples of one sequence differ from each other.
        first, second = (s[0][0][0]["C4'"][0].pos for s in structures[:2])
        assert first.dist(second) > 1e-3

    # Same seed, different batch size: the same structures.
    cfg_single = copy.deepcopy(cfg)
    cfg_single.batch_size = 1
    cfg_single.output_dir = str(tmp_path / "predictions_single")
    cfg_single.output_format = "pdb"
    run_inference(cfg_single)
    for i in range(3):
        batched = gemmi.read_structure(str(out / "rna_1" / f"Sample_{i}.pdb"))
        single = gemmi.read_structure(
            str(tmp_path / "predictions_single" / "rna_1" / f"Sample_{i}.pdb")
        )
        for res_a, res_b in zip(batched[0][0], single[0][0], strict=True):
            assert res_a["C4'"][0].pos.dist(res_b["C4'"][0].pos) < 0.05


def test_run_inference_rejects_unknown_format_before_loading(tmp_path):
    cfg = InferenceConfig(
        ckpt_path="does/not/exist.ckpt",
        fasta_path="does/not/exist.fasta",
        output_dir=str(tmp_path),
        num_samples=1,
        batch_size=1,
        seed=0,
        output_format="xyz",
        device="cpu",
    )
    with pytest.raises(ValueError, match="Unknown output_format"):
        run_inference(cfg)


def test_inference_config_matches_schema():
    with initialize_config_dir(
        config_dir=str(PROJECT_ROOT / "configs"), version_base=None
    ):
        cfg = compose(
            config_name="inference",
            overrides=["ckpt_path=model.ckpt", "fasta_path=in.fasta"],
        )
    merged = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(InferenceConfig), cfg)
    )
    assert isinstance(merged, InferenceConfig)
    assert merged.num_timesteps is None
    assert merged.output_format == "pdb"

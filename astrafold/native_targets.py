"""
Benchmark targets given as a FASTA file plus native PDB structures (e.g. CASP15).

Experimental structures rarely match the target sequence one to one: residues can be
unresolved, the numbering can be shifted or closed up over gaps, and a target can
have several native conformations or chains. Each native chain is therefore mapped
onto the target sequence by a global sequence alignment, and residues that are
missing from the native are masked out.

Natives of target <id> are found as <natives_dir>/<id>*.{pdb,cif} and
<natives_dir>/<id>/*.{pdb,cif}, sorted by name; the first is the primary native. Of
each file the first chain containing nucleotides is used (one chain is predicted), so
natives taken from multi-chain entries should be saved as single-chain files (see
scripts/download_rna3db_test.py).
"""

from dataclasses import dataclass, field
from pathlib import Path

import gemmi
import numpy as np
from Bio import Align

from astrafold.data.sequence import read_fasta

NUCLEOTIDES = "ACGU"
NATIVE_EXTENSIONS = (".pdb", ".cif")


@dataclass
class Native:
    """One native structure, on the residue indices of the target sequence."""

    label: str  # file name without the extension
    c1: np.ndarray  # [N, 3] C1' coordinates; zeros where unresolved
    mask: np.ndarray  # [N] True where the residue is resolved in the native
    # native residue (seqid number + insertion code) for every resolved index
    residues: dict[int, tuple[int, str]] = field(repr=False)
    path: Path = field(repr=False)
    chain: str = ""


@dataclass
class Target:
    name: str
    sequence: str
    natives: list[Native]


def _chain_sequence(chain: gemmi.Chain) -> tuple[str, list[gemmi.Residue]]:
    residues = [r for r in chain if r.name.strip() in NUCLEOTIDES]
    return "".join(r.name.strip() for r in residues), residues


def _aligner() -> Align.PairwiseAligner:
    # Free end gaps: natives often miss residues at the termini.
    return Align.PairwiseAligner(
        mode="global",
        match_score=2,
        mismatch_score=-3,
        open_gap_score=-5,
        extend_gap_score=-0.5,
        end_gap_score=0,
    )


def map_native(sequence: str, path: Path, min_identity: float = 0.9) -> Native:
    """
    Maps the first nucleotide chain of `path` onto `sequence`. Raises a ValueError
    when fewer than `min_identity` of the native's residues align to identical
    residues of the target, which means the native belongs to another sequence.
    """
    structure = gemmi.read_structure(str(path))
    structure.remove_hydrogens()
    for chain in structure[0]:
        native_seq, residues = _chain_sequence(chain)
        if native_seq:
            break
    else:
        raise ValueError(f"{path}: no nucleotide chain")

    alignment = _aligner().align(sequence, native_seq)[0]
    n = len(sequence)
    c1 = np.zeros((n, 3), dtype=np.float32)
    mask = np.zeros(n, dtype=bool)
    numbering: dict[int, tuple[int, str]] = {}
    identical = 0
    for (t_start, t_end), (n_start, _) in zip(*alignment.aligned, strict=True):
        for k in range(t_end - t_start):
            i, residue = t_start + k, residues[n_start + k]
            if sequence[i] != native_seq[n_start + k]:
                continue
            identical += 1
            atom = residue.find_atom("C1'", "*")
            if atom is None:
                continue
            c1[i] = (atom.pos.x, atom.pos.y, atom.pos.z)
            mask[i] = True
            numbering[i] = (residue.seqid.num, residue.seqid.icode)

    if identical < min_identity * len(native_seq):
        raise ValueError(
            f"{path}: only {identical}/{len(native_seq)} native residues match the "
            "target sequence"
        )
    return Native(path.stem, c1, mask, numbering, path, chain.name)


def find_native_files(natives_dir: Path, name: str) -> list[Path]:
    files = [
        f
        for ext in NATIVE_EXTENSIONS
        for f in [
            *natives_dir.glob(f"{name}*{ext}"),
            *(natives_dir / name).glob(f"*{ext}"),
        ]
    ]
    return sorted(set(files), key=lambda p: p.name)


def load_targets(fasta_path: str | Path, natives_dir: str | Path) -> list[Target]:
    """Targets of the FASTA file (record id = target name) with their natives."""
    natives_dir = Path(natives_dir)
    targets = []
    for name, sequence in read_fasta(fasta_path):
        files = find_native_files(natives_dir, name)
        if not files:
            raise FileNotFoundError(f"No native structure for {name} in {natives_dir}")
        natives = [map_native(sequence, f) for f in files]
        targets.append(Target(name, sequence, natives))
    return targets


def write_renumbered(native: Native, out_path: Path) -> None:
    """
    Writes the native chain with residue i of the target sequence numbered i + 1
    (unmapped residues dropped), so residue-index based tools such as
    `USalign -TMscore 1` can compare it with a prediction of the target sequence.
    """
    structure = gemmi.read_structure(str(native.path))
    structure.remove_hydrogens()
    structure.remove_alternative_conformations()
    source = structure[0][native.chain]
    by_id = {(r.seqid.num, r.seqid.icode): r for r in source}

    new = gemmi.Structure()
    model = gemmi.Model("1")
    chain = gemmi.Chain("A")
    for i in sorted(native.residues):
        residue = by_id[native.residues[i]].clone()
        residue.seqid = gemmi.SeqId(i + 1, " ")
        chain.add_residue(residue)
    model.add_chain(chain)
    new.add_model(model)
    new.setup_entities()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    new.write_pdb(str(out_path))

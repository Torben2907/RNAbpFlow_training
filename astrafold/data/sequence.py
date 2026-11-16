"""Model inputs that only depend on the RNA sequence, as needed for inference."""

from pathlib import Path

import numpy as np
import torch
from Bio import SeqIO
from torch import Tensor

from astrafold.data import base_constants as rc

# Channel order of the "onehot" feature in the training pickles. It differs from the
# order of `aatype` (base_constants.restypes: A, G, C, U).
ONEHOT_ORDER = "AUGC"


def _atom28_tables() -> tuple[np.ndarray, np.ndarray]:
    """
    Per residue type, for each of the 28 atom types: the index of that atom in the
    residue's dense atom23 layout (0 if the residue has no such atom), and whether the
    residue has it.
    """
    atom28_to_atom23, atom28_mask = [], []
    for restype in rc.restypes:
        atom23_names = rc.restype_name_to_atom23_names[rc.restype_1to3[restype]]
        index = {name: i for i, name in enumerate(atom23_names) if name}
        atom28_to_atom23.append([index.get(name, 0) for name in rc.atom_types])
        atom28_mask.append([float(name in index) for name in rc.atom_types])
    return np.array(atom28_to_atom23), np.array(atom28_mask, dtype=np.float32)


_ATOM28_TO_ATOM23, _ATOM28_MASK = _atom28_tables()


def normalize_sequence(sequence: str) -> str:
    """Upper-cases the sequence and reads T as U. Raises on anything but A, C, G, U."""
    sequence = sequence.strip().upper().replace("T", "U")
    unknown = sorted(set(sequence) - set(ONEHOT_ORDER))
    if unknown:
        raise ValueError(
            f"Unsupported nucleotides {unknown}: only A, C, G and U (or T) are allowed"
        )
    if not sequence:
        raise ValueError("Empty sequence")
    return sequence


def sequence_features(sequence: str) -> dict[str, Tensor]:
    """
    Returns the features of an RNA sequence of length N:
        onehot           : [N, 4] one-hot sequence, the model input (order A, U, G, C)
        aatype           : [N] residue type index into base_constants.restypes
        atom28_to_atom23 : [N, 28] gather indices from the atom23 to the atom28 layout
        atom28_mask      : [N, 28] 1 where the residue has that atom
    """
    sequence = normalize_sequence(sequence)
    onehot_idx = torch.tensor([ONEHOT_ORDER.index(nt) for nt in sequence])
    aatype = torch.tensor([rc.restype_order[nt] for nt in sequence])
    return {
        "onehot": torch.nn.functional.one_hot(onehot_idx, num_classes=4).float(),
        "aatype": aatype,
        "atom28_to_atom23": torch.from_numpy(_ATOM28_TO_ATOM23)[aatype],
        "atom28_mask": torch.from_numpy(_ATOM28_MASK)[aatype],
    }


def read_fasta(path: str | Path) -> list[tuple[str, str]]:
    """Returns the (id, sequence) pairs of a FASTA file."""
    records = [(record.id, str(record.seq)) for record in SeqIO.parse(path, "fasta")]
    if not records:
        raise ValueError(f"No FASTA records found in {path}")
    return records

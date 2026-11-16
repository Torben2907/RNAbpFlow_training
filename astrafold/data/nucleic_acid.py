"""
Writes all-atom RNA structures as PDB or mmCIF files.

The PDB writer is adapted from NuFold (https://github.com/kiharalab/NuFold).
"""

from pathlib import Path

import gemmi
import numpy as np

from astrafold.data import base_constants as rc

OUTPUT_FORMATS = ("pdb", "mmcif", "both")


def to_pdb(
    aatype: np.ndarray,
    atom_positions: np.ndarray,
    atom_mask: np.ndarray,
    chain_id: str = "A",
) -> str:
    """
    Formats a single RNA chain as a PDB string. Residues are numbered from 1.

    Params:
        aatype         : [N] residue type index into base_constants.restypes
        atom_positions : [N, 28, 3] coordinates in Angstrom (base_constants.atom_types)
        atom_mask      : [N, 28] atoms with a mask below 0.5 are left out
    """
    if np.any(aatype >= rc.restype_num) or np.any(aatype < 0):
        raise ValueError("Invalid residue types")

    lines = []
    atom_index = 1
    for i, restype in enumerate(aatype):
        res_name = rc.restype_1to3[rc.restypes[restype]]
        for atom_name, pos, mask in zip(
            rc.atom_types, atom_positions[i], atom_mask[i], strict=True
        ):
            if mask < 0.5:
                continue
            name = atom_name if len(atom_name) == 4 else f" {atom_name}"
            element = atom_name[0]  # P, O, C or N
            # PDB is a columnar format, every space matters here!
            lines.append(
                f"{'ATOM':<6}{atom_index:>5} {name:<4}{'':>1}"
                f"{res_name:>3} {chain_id:>1}"
                f"{i + 1:>4}{'':>1}   "
                f"{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}"
                f"{1.0:>6.2f}{0.0:>6.2f}          "
                f"{element:>2}{'':>2}"
            )
            atom_index += 1

    last_res_name = rc.restype_1to3[rc.restypes[aatype[-1]]]
    lines.append(
        f"{'TER':<6}{atom_index:>5}      {last_res_name:>3} "
        f"{chain_id:>1}{len(aatype):>4}"
    )
    lines.append("END")
    lines.append("")
    return "\n".join(lines)


def write_structure(
    path_stem: str | Path,
    aatype: np.ndarray,
    atom_positions: np.ndarray,
    atom_mask: np.ndarray,
    output_format: str = "pdb",
) -> list[Path]:
    """
    Writes `<path_stem>.pdb`, `<path_stem>.cif` or both, depending on `output_format`
    ("pdb", "mmcif" or "both"). Returns the written paths.
    """
    if output_format not in OUTPUT_FORMATS:
        raise ValueError(
            f"Unknown output format {output_format!r}, choose from {OUTPUT_FORMATS}"
        )

    pdb_text = to_pdb(aatype, atom_positions, atom_mask)
    path_stem = Path(path_stem)
    written = []
    if output_format in ("pdb", "both"):
        pdb_path = path_stem.parent / f"{path_stem.name}.pdb"
        pdb_path.write_text(pdb_text)
        written.append(pdb_path)
    if output_format in ("mmcif", "both"):
        cif_path = path_stem.parent / f"{path_stem.name}.cif"
        structure = gemmi.read_pdb_string(pdb_text)
        structure.setup_entities()
        structure.make_mmcif_document().write_file(str(cif_path))
        written.append(cif_path)
    return written

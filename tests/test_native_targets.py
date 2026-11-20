from pathlib import Path

import gemmi
import numpy as np
import pytest

from astrafold.data.sequence import read_fasta
from astrafold.native_targets import (
    find_native_files,
    load_targets,
    map_native,
    write_renumbered,
)

CASP15 = Path(__file__).resolve().parents[1] / "benchmarks" / "CASP15"
FASTA = CASP15 / "CASP15_test.fasta"
NATIVES = CASP15 / "natives"
SEQUENCES = dict(read_fasta(FASTA)) if FASTA.exists() else {}

pytestmark = pytest.mark.skipif(
    not NATIVES.exists(), reason="CASP15 natives missing: scripts/download_casp15.sh"
)


def test_unresolved_first_residue_is_masked() -> None:
    # R1117: the native starts at residue 2 of the 30-nt target
    native = map_native(SEQUENCES["R1117"], NATIVES / "R1117.pdb")
    assert native.mask.sum() == 29
    assert not native.mask[0] and native.mask[1:].all()
    assert native.residues[1][0] == 2


def test_shifted_and_closed_up_numbering_is_aligned() -> None:
    # R1116: the native numbering is shifted by one against the target (native #1 is
    # target residue 2) and jumps from #46 to #56 over an unresolved 9-nt loop
    # (15 A between the C1' atoms), i.e. target residues 48-56; the last residue
    # of the target is unresolved too
    native = map_native(SEQUENCES["R1116"], NATIVES / "R1116_PV_CL_Final.pdb")
    assert native.mask.sum() == 146
    missing = np.flatnonzero(~native.mask) + 1
    assert list(missing) == [1, *range(48, 57), 157]
    assert native.residues[46][0] == 46 and native.residues[56][0] == 56


def test_wrong_sequence_is_rejected() -> None:
    with pytest.raises(ValueError, match="match the target sequence"):
        map_native(SEQUENCES["R1107"], NATIVES / "R1117.pdb")


def test_first_chain_and_all_conformations_are_used() -> None:
    targets = {t.name: t for t in load_targets(FASTA, NATIVES)}
    assert targets["R1108"].natives[0].chain == "0"  # the native is a dimer
    assert len(targets["R1149"].natives) == 10  # NMR-like ensemble in a subfolder
    assert [n.label for n in targets["R1136"].natives] == ["R1136v1", "R1136v2"]
    assert len(find_native_files(NATIVES, "R1138")) == 2


def test_renumbered_native_matches_target_indices(tmp_path: Path) -> None:
    native = map_native(SEQUENCES["R1116"], NATIVES / "R1116_PV_CL_Final.pdb")
    out = tmp_path / "R1116.pdb"
    write_renumbered(native, out)
    chain = gemmi.read_structure(str(out))[0][0]
    for residue in chain:
        i = residue.seqid.num - 1
        assert residue.name.strip() == SEQUENCES["R1116"][i]
        pos = residue.find_atom("C1'", "*").pos
        np.testing.assert_allclose([pos.x, pos.y, pos.z], native.c1[i], atol=1e-3)
    assert len(chain) == 146

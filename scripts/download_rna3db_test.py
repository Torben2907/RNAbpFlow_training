"""
Downloads the native structures of the RNA3DB test set (the 48 chains of
benchmarks/RNA3DB/RNA3DB_test.fasta, the test list of RNAbpFlow, from the 2024-04-26
RNA3DB release) from the RCSB PDB. The structures are not part of the repository.

Each FASTA id is <pdb id>_<chain>. The chain is cut out of the entry (first model) and
saved as a single-chain mmCIF file <out>/<id>.cif, so that astrafold.native_targets
picks the right chain. The chain is looked up by its author chain id first and its
label (mmCIF) id second, and kept only if its sequence aligns to the FASTA sequence
(see astrafold.native_targets.map_native).

    python3 scripts/download_rna3db_test.py [--out benchmarks/RNA3DB/natives]
"""

import argparse
import gzip
import sys
import urllib.request
from pathlib import Path

import gemmi

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from astrafold.data.sequence import read_fasta  # noqa: E402
from astrafold.native_targets import map_native  # noqa: E402

URL = "https://files.rcsb.org/download/{}.cif.gz"


def fetch_entry(pdb_id: str, cache: Path) -> gemmi.Structure:
    path = cache / f"{pdb_id}.cif.gz"
    if not path.exists():
        with urllib.request.urlopen(URL.format(pdb_id.upper()), timeout=120) as r:
            path.write_bytes(r.read())
    doc = gemmi.cif.read_string(gzip.decompress(path.read_bytes()).decode())
    structure = gemmi.make_structure_from_block(doc.sole_block())
    structure.remove_hydrogens()
    structure.remove_alternative_conformations()
    return structure


def extract_chain(structure: gemmi.Structure, chain_id: str) -> gemmi.Structure:
    """Single-chain structure of the first model; author id first, then label id."""
    model = structure[0]
    chain = model.find_chain(chain_id)
    if chain is None:
        subchain = model.get_subchain(chain_id)
        if len(subchain) == 0:
            raise KeyError(f"no chain {chain_id}")
        chain = gemmi.Chain(chain_id)
        for residue in subchain:
            chain.add_residue(residue)
    new = gemmi.Structure()
    new.name = structure.name
    new.cell = structure.cell
    new.spacegroup_hm = structure.spacegroup_hm
    new_model = gemmi.Model("1")
    new_model.add_chain(chain)
    new.add_model(new_model)
    new.setup_entities()
    return new


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--fasta", default=str(PROJECT_ROOT / "benchmarks/RNA3DB/RNA3DB_test.fasta")
    )
    parser.add_argument(
        "--out", default=str(PROJECT_ROOT / "benchmarks/RNA3DB/natives")
    )
    args = parser.parse_args()
    out = Path(args.out)
    cache = out / ".entries"
    cache.mkdir(parents=True, exist_ok=True)

    records = read_fasta(args.fasta)
    ids = [name for name, _ in records]
    clashes = [a for a in ids for b in ids if a != b and b.startswith(a)]
    if clashes:
        raise SystemExit(f"Ids that are prefixes of other ids: {clashes}")

    failed = []
    for name, sequence in records:
        target = out / f"{name}.cif"
        if target.exists():
            continue
        pdb_id, chain_id = name.split("_", 1)
        try:
            chain = extract_chain(fetch_entry(pdb_id, cache), chain_id)
            tmp = out / f"{name}.tmp.cif"
            chain.make_mmcif_document().write_file(str(tmp))
            native = map_native(sequence, tmp)  # raises if the sequence differs
            tmp.rename(target)
            print(f"{name}: {native.mask.sum()}/{len(sequence)} residues resolved")
        except Exception as e:  # report every failure, then exit non-zero
            failed.append(name)
            print(f"{name}: FAILED ({e})", file=sys.stderr)
            (out / f"{name}.tmp.cif").unlink(missing_ok=True)

    if failed:
        raise SystemExit(f"{len(failed)} targets failed: {failed}")
    print(f"Wrote {len(records)} native structures to {out}")


if __name__ == "__main__":
    main()

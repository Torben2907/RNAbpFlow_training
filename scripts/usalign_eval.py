"""
Scores the structures saved by `compare_samplers.py run --save-structures` with
US-align (https://github.com/pylelab/USalign), as used for RNA benchmarks such as
CASP15: TM-score normalized by the length of the native, and RMSD.

Two alignments per prediction and native:
  - `tm`, `rmsd`: residue-index superposition (`-TMscore 1`), possible because the
    natives in <out>/natives/ are renumbered onto the target sequence;
  - `tm_align`, `rmsd_align`: US-align's default sequence-independent alignment.
US-align uses the C3' atoms of RNA (the metrics of compare_samplers use C1').

    python3 scripts/usalign_eval.py --out results/casp15_ft [--usalign PATH]

Writes <out>/usalign.csv (one row per sample and native) and
<out>/usalign_summary.md. The first native of a target (by file name) is the
primary one; `any` columns take the best over all natives of the target.
"""

import argparse
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_USALIGN = Path.home() / "tools" / "USalign" / "USalign"


def usalign(binary: str, pred: Path, native: Path, by_index: bool) -> tuple:
    """(TM-score normalized by the native, RMSD, aligned length)."""
    cmd = [binary, str(pred), str(native), "-mol", "RNA", "-outfmt", "2"]
    if by_index:
        cmd += ["-TMscore", "1"]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    fields = [line for line in out.splitlines() if not line.startswith("#")][0]
    fields = fields.split("\t")
    # columns: chain1 chain2 TM1 TM2 RMSD ID1 ID2 IDali L1 L2 Lali
    return float(fields[3]), float(fields[4]), int(fields[10])


def score_all(out: Path, binary: str, workers: int) -> pd.DataFrame:
    jobs = []
    for pred in sorted((out / "structures").glob("*/*/Sample_*.pdb")):
        sampler, target = pred.parent.parent.name, pred.parent.name
        natives = sorted((out / "natives" / target).glob("*.pdb"))
        if not natives:
            raise FileNotFoundError(f"No natives for {target} in {out / 'natives'}")
        for rank, native in enumerate(natives):
            jobs.append((sampler, target, pred, native, rank))

    def run(job):
        sampler, target, pred, native, rank = job
        tm, rmsd, lali = usalign(binary, pred, native, by_index=True)
        tm_a, rmsd_a, _ = usalign(binary, pred, native, by_index=False)
        return {
            "sampler": sampler,
            "pdb_name": target,
            "seed": int(pred.stem.removeprefix("Sample_")),
            "native": native.stem,
            "primary": rank == 0,
            "tm": tm,
            "rmsd": rmsd,
            "aligned": lali,
            "tm_align": tm_a,
            "rmsd_align": rmsd_a,
        }

    with ThreadPoolExecutor(workers) as pool:
        rows = list(pool.map(run, jobs))
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, out: Path) -> str:
    samples = pd.concat(pd.read_csv(p) for p in sorted(out.glob("samples_*.csv")))
    scores = samples[["pdb_name", "sampler", "seed", "self_consistency", "length"]]

    primary = df[df.primary].merge(scores, on=["pdb_name", "sampler", "seed"])
    best_any = df.groupby(["pdb_name", "sampler", "seed"])[["tm", "tm_align"]].max()
    best_any_rmsd = df.groupby(["pdb_name", "sampler", "seed"]).rmsd.min()
    primary = primary.join(best_any.add_suffix("_any"), on=["pdb_name", "sampler", "seed"])
    primary = primary.join(best_any_rmsd.rename("rmsd_any"), on=["pdb_name", "sampler", "seed"])

    samplers = list(dict.fromkeys(primary.sampler))
    targets = list(dict.fromkeys(samples.pdb_name))
    fmt = "{:.3f}".format

    def per_target(d: pd.DataFrame) -> dict:
        top1 = (
            d.loc[d.self_consistency.idxmax()]
            if d.self_consistency.notna().all()
            else None
        )
        return {
            "tm_mean": d.tm.mean(),
            "tm_best": d.tm.max(),
            "tm_top1": top1.tm if top1 is not None else float("nan"),
            "tm_first": d.loc[d.seed.idxmin()].tm,
            "rmsd_mean": d.rmsd.mean(),
            "rmsd_best": d.rmsd.min(),
            "rmsd_top1": top1.rmsd if top1 is not None else float("nan"),
            "rmsd_first": d.loc[d.seed.idxmin()].rmsd,
            "tm_align_mean": d.tm_align.mean(),
            "tm_any_best": d.tm_any.max(),
            "rmsd_any_best": d.rmsd_any.min(),
        }

    table = pd.DataFrame(
        [
            {"pdb_name": t, "sampler": s, "length": d.length.iloc[0]} | per_target(d)
            for (t, s), d in primary.groupby(["pdb_name", "sampler"], sort=False)
        ]
    )
    k = primary.groupby(["pdb_name", "sampler"]).size().max()
    lines = [
        f"# US-align scores ({out.name})",
        "",
        (
            f"{len(targets)} targets, {k} samples per target and sampler. TM-score "
            "normalized by the native length, residue-index superposition "
            "(`USalign -mol RNA -TMscore 1`, C3' atoms) against the first native of "
            "each target. *first* = sample of the first seed (a single prediction), "
            "*top-1* = highest self-consistency score (hybrid only), *best* = "
            "best of all samples (oracle). `tm_align` = sequence-independent "
            "alignment; *any* = best over all natives of a target."
        ),
        "",
        "## Mean over targets",
        "",
        "| Sampler | TM first | TM top-1 | TM mean | TM best | RMSD first | RMSD top-1 "
        "| RMSD mean | RMSD best | TM-align mean | TM best (any native) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in samplers:
        t = table[table.sampler == s]
        lines.append(
            f"| {s} | {fmt(t.tm_first.mean())} | {fmt(t.tm_top1.mean())} "
            f"| {fmt(t.tm_mean.mean())} | {fmt(t.tm_best.mean())} "
            f"| {t.rmsd_first.mean():.2f} | {t.rmsd_top1.mean():.2f} "
            f"| {t.rmsd_mean.mean():.2f} | {t.rmsd_best.mean():.2f} "
            f"| {fmt(t.tm_align_mean.mean())} | {fmt(t.tm_any_best.mean())} |"
        )
    lines += [
        "",
        "## Per target",
        "",
        "| Target | N | Sampler | TM first | TM top-1 | TM mean | TM best | RMSD first "
        "| RMSD top-1 | RMSD mean | RMSD best | TM best (any) | RMSD best (any) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for t in targets:
        for s in samplers:
            r = table[(table.pdb_name == t) & (table.sampler == s)]
            if r.empty:
                continue
            r = r.iloc[0]
            lines.append(
                f"| {t} | {r.length} | {s} | {fmt(r.tm_first)} | {fmt(r.tm_top1)} "
                f"| {fmt(r.tm_mean)} | {fmt(r.tm_best)} | {r.rmsd_first:.2f} "
                f"| {r.rmsd_top1:.2f} | {r.rmsd_mean:.2f} | {r.rmsd_best:.2f} "
                f"| {fmt(r.tm_any_best)} | {r.rmsd_any_best:.2f} |"
            )
    return "\n".join(lines).replace("| nan ", "| – ") + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--usalign",
        default=os.environ.get("USALIGN")
        or shutil.which("USalign")
        or str(DEFAULT_USALIGN),
    )
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()
    os.chdir(PROJECT_ROOT)
    if not Path(args.usalign).exists():
        raise SystemExit(f"US-align not found at {args.usalign} (set --usalign)")

    out = Path(args.out)
    df = score_all(out, args.usalign, args.workers)
    df.to_csv(out / "usalign.csv", index=False)
    text = summarize(df, out)
    (out / "usalign_summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()

"""
Compares the flow and the hybrid sampler on held-out RNAs.

Every RNA is sampled `--num-samples` times per sampler, with the same seeds (and
therefore the same initial noise) for all samplers, and each sample is compared
with the ground truth on the C1' atoms: RMSD, TM-score and lDDT, plus the accuracy
of the generated contact tokens.

The RNAs come either from a metadata CSV of processed features (default: the
RNA3DB validation split) or, with `--fasta` and `--natives`, from a FASTA file plus
native PDB structures, e.g. a CASP benchmark (see astrafold/native_targets.py).
With several natives per target (conformations, NMR models), the metrics are
reported against the first native and as the best over all natives (`*_any`).

    # one shard per GPU (run from anywhere; paths are relative to the project root)
    python3 scripts/compare_samplers.py run --ckpt <ckpt> --out results/compare \\
        --shard 0 --num-shards 4 --device cuda:0
    # CASP15 benchmark, keeping the predicted structures (e.g. for US-align)
    python3 scripts/compare_samplers.py run --ckpt <ckpt> --out results/casp15 \\
        --fasta benchmarks/CASP15/CASP15_test.fasta \\
        --natives benchmarks/CASP15/natives --save-structures \\
        --shard 0 --num-shards 4 --device cuda:0
    # after all shards are done
    python3 scripts/compare_samplers.py summarize --out results/compare

`run` writes one row per sample to <out>/samples_<shard>.csv (appended after every
RNA and sampler, so finished work survives an interruption and is skipped on a
rerun); `summarize` combines the shards into <out>/summary.md. With natives, the
work is split over the shards by its estimated cost (long RNAs and the hybrid
sampler dominate), and `--save-structures` also writes the natives renumbered onto
the target sequence to <out>/natives/.
"""

import argparse
import csv
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, wilcoxon

from astrafold.config.hydra_schema import HybridSamplerConfig
from astrafold.data import base_constants as rc
from astrafold.data.contacts import make_contact_tokens
from astrafold.data.nucleic_acid import write_structure
from astrafold.data.pdb_na_dataset_base import PDBNABaseDataset
from astrafold.inference import Sampler
from astrafold.metrics import kabsch_rmsd, lddt, tm_score
from astrafold.native_targets import load_targets, write_renumbered

PROJECT_ROOT = Path(__file__).resolve().parents[1]
C1 = rc.atom_order["C1'"]

COLUMNS = [
    "pdb_name",
    "length",
    "sampler",
    "seed",
    "rmsd",
    "tm_score",
    "lddt",
    "token_accuracy",
    "rmsd_any",
    "tm_score_any",
    "lddt_any",
    "num_natives",
    "self_consistency",
    "num_forced",
    "num_model_calls",
    "seconds",
]

# Samples per forward pass are limited to keep batch * N^2 (the pair features) in
# memory: 10 samples of 374 nt fit in ~9 GiB, 10 of 720 nt do not fit in 40 GiB.
MAX_BATCH_RES_SQUARED = 1_500_000

# Model calls per sample, for splitting the work over shards (hybrid: measured on
# the RNA3DB validation split, about 0.84 / eps calls per residue).
FLOW_CALLS = 50


@dataclass
class Example:
    """One RNA with its ground truth: C1' coordinates and mask per native."""

    name: str
    sequence: str
    natives: list[tuple[np.ndarray, np.ndarray]]  # ([N, 3], [N] bool), primary first
    tokens: np.ndarray  # [N] contact tokens of the primary native


def _csv_examples(dataset: PDBNABaseDataset, indices: list[int]) -> dict[int, Example]:
    examples = {}
    for idx in indices:
        example = {k: torch.as_tensor(v) for k, v in dataset[idx].items()}
        examples[idx] = Example(
            name=dataset.csv.iloc[idx]["pdb_name"],
            sequence="".join(rc.restypes[a] for a in example["aatype"].tolist()),
            natives=[
                (example["trans_1"].numpy(), example["res_mask"].bool().numpy())
            ],
            tokens=example["contact_tokens"].numpy(),
        )
    return examples


def _native_examples(fasta: str, natives_dir: str, data_cfg) -> list[Example]:
    examples = []
    for target in load_targets(fasta, natives_dir):
        primary = target.natives[0]
        tokens = make_contact_tokens(
            torch.from_numpy(primary.c1),
            torch.from_numpy(primary.mask),
            data_cfg.contact_distance_threshold,
            data_cfg.contact_minimum_sequence_separation,
        ).numpy()
        examples.append(
            Example(
                target.name,
                target.sequence,
                [(n.c1, n.mask) for n in target.natives],
                tokens,
            )
        )
    return examples


def _cost(length: int, sampler: str, args) -> float:
    """Relative cost of sampling one RNA: model calls times N^2."""
    if sampler == "flow":
        calls = args.num_timesteps or FLOW_CALLS
    else:
        eps = float(sampler.removeprefix("hybrid_eps"))
        calls = 0.84 / eps * length
    return calls * length**2


def _assign_shards(units: list[tuple[int, str, int]], num_shards: int, args):
    """Longest-processing-time-first split of (example, sampler, length) units."""
    loads = [0.0] * num_shards
    shard_of = {}
    for unit in sorted(units, key=lambda u: -_cost(u[2], u[1], args)):
        shard = loads.index(min(loads))
        loads[shard] += _cost(unit[2], unit[1], args)
        shard_of[unit[:2]] = shard
    return shard_of


def _score(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> tuple:
    """RMSD, TM-score and lDDT of one sample [N, 3] against one native."""
    rmsd = kabsch_rmsd(
        torch.from_numpy(pred)[None],
        torch.from_numpy(gt)[None],
        torch.from_numpy(mask)[None],
    )[0].item()
    return rmsd, tm_score(pred[mask], gt[mask]), lddt(pred[mask], gt[mask])


def _sampler_specs(args) -> dict[str, tuple[str, HybridSamplerConfig | None]]:
    """Sampler variants by name: flow, and one hybrid per confidence tolerance."""
    specs: dict[str, tuple[str, HybridSamplerConfig | None]] = {}
    if "flow" in args.samplers:
        specs["flow"] = ("flow", None)
    if "hybrid" in args.samplers:
        for eps in args.hybrid_eps:
            cfg = HybridSamplerConfig(eps, args.tokens_per_round, args.remask_fraction)
            specs[f"hybrid_eps{eps:g}"] = ("hybrid", cfg)
    return specs


def run(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / f"samples_{args.shard}.csv"
    done = set()
    if rows_path.exists():
        previous = pd.read_csv(rows_path)
        done = set(zip(previous.pdb_name, previous.sampler, strict=True))

    specs = _sampler_specs(args)
    samplers = {
        name: Sampler(args.ckpt, args.device, args.num_timesteps, kind, hybrid)
        for name, (kind, hybrid) in specs.items()
    }
    data_cfg = next(iter(samplers.values()))._module.hparams.cfg.data_cfg

    # Work units: (example, sampler) pairs of this shard, in example order.
    if args.fasta is not None:
        examples = dict(
            enumerate(_native_examples(args.fasta, args.natives, data_cfg))
        )
        units = [
            (i, name, len(ex.sequence))
            for i, ex in examples.items()
            for name in samplers
        ]
        shard_of = _assign_shards(units, args.num_shards, args)
        work = [(i, name) for i, name, _ in units if shard_of[i, name] == args.shard]
        if args.save_structures:
            _write_natives(args.fasta, args.natives, out / "natives")
    else:
        dataset = PDBNABaseDataset(data_cfg, is_training=False, csv_path=args.val_csv)
        indices = list(range(len(dataset)))[args.shard :: args.num_shards]
        examples = _csv_examples(dataset, indices)
        work = [(i, name) for i in indices for name in samplers]
    if args.limit is not None:
        keep = list(dict.fromkeys(i for i, _ in work))[: args.limit]
        work = [(i, name) for i, name in work if i in keep]
    seeds = [args.seed + i for i in range(args.num_samples)]

    write_header = not rows_path.exists()
    with open(rows_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if write_header:
            writer.writeheader()
        for count, (idx, name) in enumerate(work, start=1):
            ex = examples[idx]
            if (ex.name, name) in done:
                continue
            n = len(ex.sequence)
            batch_size = max(1, min(len(seeds), MAX_BATCH_RES_SQUARED // n**2))
            start = time.time()
            chunks = [
                samplers[name].sample(ex.sequence, seeds[k : k + batch_size])
                for k in range(0, len(seeds), batch_size)
            ]
            seconds = (time.time() - start) / len(seeds)
            result = {
                key: np.concatenate([c[key] for c in chunks])
                for key in chunks[0]
                if key not in ("atom_mask", "aatype")
            }
            pred = result["atom_positions"][:, :, C1].astype(np.float32)
            primary_mask = ex.natives[0][1]

            for j, seed in enumerate(seeds):
                scores = [_score(pred[j], gt, mask) for gt, mask in ex.natives]
                tokens = result["contact_tokens"][j]
                row = {
                    "pdb_name": ex.name,
                    "length": n,
                    "sampler": name,
                    "seed": seed,
                    "rmsd": scores[0][0],
                    "tm_score": scores[0][1],
                    "lddt": scores[0][2],
                    "token_accuracy": (tokens == ex.tokens)[primary_mask].mean(),
                    "rmsd_any": min(s[0] for s in scores),
                    "tm_score_any": max(s[1] for s in scores),
                    "lddt_any": max(s[2] for s in scores),
                    "num_natives": len(ex.natives),
                    "seconds": seconds,
                }
                for key in ("self_consistency", "num_forced", "num_model_calls"):
                    if key in result:
                        row[key] = result[key][j].item()
                writer.writerow(row)
                if args.save_structures:
                    target_dir = out / "structures" / name / ex.name
                    target_dir.mkdir(parents=True, exist_ok=True)
                    write_structure(
                        target_dir / f"Sample_{seed}",
                        chunks[0]["aatype"],
                        result["atom_positions"][j],
                        chunks[0]["atom_mask"],
                        "pdb",
                    )
            f.flush()
            print(
                f"[shard {args.shard}] {count}/{len(work)} {ex.name} (N={n}) "
                f"{name}: {seconds:.1f} s / sample",
                flush=True,
            )


def _write_natives(fasta: str, natives_dir: str, out: Path) -> None:
    """Natives renumbered onto the target sequence, for residue-index tools."""
    for target in load_targets(fasta, natives_dir):
        for native in target.natives:
            path = out / target.name / f"{native.label}.pdb"
            if not path.exists():
                write_renumbered(native, path)


def _per_target_table(df: pd.DataFrame, samplers: list[str]) -> list[str]:
    """Per RNA and sampler: mean and best-of-K TM-score and RMSD, and top-1."""
    multi = "num_natives" in df and (df.num_natives > 1).any()
    lines = [
        "",
        "## Per target",
        "",
        (
            "Mean / best-of-K over the samples; top-1 = sample with the highest "
            "self-consistency score (hybrid only)."
            + (
                " `any` = best over all native conformations / models; the other "
                "columns use the first native."
                if multi
                else ""
            )
        ),
        "",
        "| Target | N | Sampler | TM mean | TM best | TM top-1 | RMSD mean | RMSD best "
        + ("| TM best (any) | RMSD best (any) " if multi else "")
        + "|",
        "|---|---|---|---|---|---|---|---" + ("|---|---" if multi else "") + "|",
    ]
    for pdb_name, d in df.groupby("pdb_name", sort=False):
        for name in samplers:
            s = d[d.sampler == name]
            top1 = (
                _fmt(s.loc[s.self_consistency.idxmax()].tm_score)
                if s.self_consistency.notna().all()
                else "–"
            )
            row = (
                f"| {pdb_name} | {s.length.iloc[0]} | {name} | {_fmt(s.tm_score.mean())} "
                f"| {_fmt(s.tm_score.max())} | {top1} | {_fmt(s.rmsd.mean())} "
                f"| {_fmt(s.rmsd.min())} "
            )
            if multi:
                row += f"| {_fmt(s.tm_score_any.max())} | {_fmt(s.rmsd_any.min())} "
            lines.append(row + "|")
    return lines


def _fmt(x: float) -> str:
    return f"{x:.3f}" if abs(x) < 100 else f"{x:.0f}"


def summarize(args) -> None:
    out = Path(args.out)
    shards = sorted(out.glob("samples_*.csv"))
    if not shards:
        raise SystemExit(f"No samples_*.csv in {out}; run the `run` step first")
    df = pd.concat(pd.read_csv(p) for p in shards)
    samplers = list(dict.fromkeys(df.sampler))
    # Only RNAs that every sampler has finished are compared.
    complete = df.groupby("pdb_name").sampler.nunique() == len(samplers)
    df = df[df.pdb_name.isin(complete[complete].index)]
    if df.empty:
        raise SystemExit("No RNA has results from every sampler yet")
    num_rnas = df.pdb_name.nunique()
    lines = [
        f"# Flow vs hybrid sampler ({out.name})",
        "",
        (
            f"{num_rnas} RNAs (length {df.length.min()}-{df.length.max()}), "
            f"{df.groupby(['pdb_name', 'sampler']).size().max()} samples per RNA and "
            "sampler, same seeds for all samplers. Metrics on C1' atoms; per RNA "
            "first averaged over samples, then over RNAs (mean, median in brackets)."
        ),
        "",
        "## Quality",
        "",
        (
            "| Sampler | RMSD (Å) ↓ | best-of-K RMSD ↓ | TM-score ↑ | best-of-K TM ↑ "
            "| lDDT ↑ | token acc. ↑ | top-1 TM by score ↑ | s / sample "
            "| calls / sample | forced |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    per_rna = {}
    for name in samplers:
        d = df[df.sampler == name]
        g = d.groupby("pdb_name")
        mean = g[["rmsd", "tm_score", "lddt", "token_accuracy"]].mean()
        per_rna[name] = mean
        best_rmsd, best_tm = g.rmsd.min(), g.tm_score.max()
        if d.self_consistency.notna().all():
            top1 = d.loc[g.self_consistency.idxmax()].set_index("pdb_name").tm_score
            top1_str = _fmt(top1.mean())
        else:
            top1_str = "–"

        def cell(s: pd.Series) -> str:
            return f"{_fmt(s.mean())} ({_fmt(s.median())})"

        lines.append(
            f"| {name} | {cell(mean.rmsd)} | {cell(best_rmsd)} | {cell(mean.tm_score)} "
            f"| {cell(best_tm)} | {cell(mean.lddt)} | {cell(mean.token_accuracy)} "
            f"| {top1_str} | {_fmt(d.seconds.mean())} "
            f"| {_fmt(d.num_model_calls.mean()) if d.num_model_calls.notna().any() else '50'} "
            f"| {_fmt(d.num_forced.mean()) if d.num_forced.notna().any() else '–'} |"
        )

    if "flow" in per_rna:
        lines += [
            "",
            "## Paired comparison with flow (per RNA, mean over samples)",
            "",
            "| Sampler | metric | mean Δ vs flow | RNAs better | Wilcoxon p |",
            "|---|---|---|---|---|",
        ]
        for name in samplers:
            if name == "flow":
                continue
            for metric, higher_is_better in (
                ("rmsd", False),
                ("tm_score", True),
                ("lddt", True),
            ):
                a = per_rna[name][metric]
                b = per_rna["flow"][metric].loc[a.index]
                delta = a - b
                better = (delta > 0) if higher_is_better else (delta < 0)
                p = wilcoxon(a, b).pvalue if (delta != 0).any() else float("nan")
                lines.append(
                    f"| {name} | {metric} | {delta.mean():+.3f} "
                    f"| {better.sum()}/{len(delta)} | {p:.3g} |"
                )

    if num_rnas <= 30:
        lines += _per_target_table(df, samplers)

    hybrids = [
        n for n in samplers if df[df.sampler == n].self_consistency.notna().all()
    ]
    if hybrids:
        lines += [
            "",
            "## Does the self-consistency score rank the samples?",
            "",
            (
                "Spearman correlation between the score and the quality of the "
                "samples of one RNA, averaged over RNAs (positive = higher score, "
                "better structure)."
            ),
            "",
            "| Sampler | ρ(score, TM-score) | ρ(score, −RMSD) |",
            "|---|---|---|",
        ]
        for name in hybrids:
            d = df[df.sampler == name]
            rho_tm, rho_rmsd = [], []
            for _, r in d.groupby("pdb_name"):
                if r.self_consistency.nunique() > 1:
                    rho_tm.append(spearmanr(r.self_consistency, r.tm_score)[0])
                    rho_rmsd.append(spearmanr(r.self_consistency, -r.rmsd)[0])
            lines.append(
                f"| {name} | {np.nanmean(rho_tm):+.3f} | {np.nanmean(rho_rmsd):+.3f} |"
            )

    text = "\n".join(lines) + "\n"
    (out / "summary.md").write_text(text)
    print(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="sample and score one shard")
    p_run.add_argument("--ckpt", required=True)
    p_run.add_argument("--out", default="results/compare_samplers")
    p_run.add_argument("--val-csv", default="metadata/RNA3DB_val.csv")
    p_run.add_argument(
        "--fasta", default=None, help="targets as FASTA (with --natives), not a CSV"
    )
    p_run.add_argument("--natives", default=None, help="native PDBs of --fasta")
    p_run.add_argument(
        "--save-structures",
        action="store_true",
        help="write <out>/structures/<sampler>/<target>/Sample_<seed>.pdb",
    )
    p_run.add_argument("--samplers", nargs="+", default=["flow", "hybrid"])
    p_run.add_argument("--hybrid-eps", type=float, nargs="+", default=[0.1])
    p_run.add_argument("--tokens-per-round", type=int, default=1)
    p_run.add_argument("--remask-fraction", type=float, default=1.0)
    p_run.add_argument("--num-samples", type=int, default=5)
    p_run.add_argument("--num-timesteps", type=int, default=None)
    p_run.add_argument("--seed", type=int, default=0)
    p_run.add_argument("--device", default="auto")
    p_run.add_argument("--shard", type=int, default=0)
    p_run.add_argument("--num-shards", type=int, default=1)
    p_run.add_argument("--limit", type=int, default=None, help="RNAs per shard")

    p_sum = sub.add_parser("summarize", help="combine the shards into summary.md")
    p_sum.add_argument("--out", default="results/compare_samplers")

    args = parser.parse_args()
    if args.command == "run" and (args.fasta is None) != (args.natives is None):
        parser.error("--fasta and --natives go together")
    os.chdir(PROJECT_ROOT)  # the metadata CSVs hold paths relative to the root
    torch.set_float32_matmul_precision("high")
    {"run": run, "summarize": summarize}[args.command](args)


if __name__ == "__main__":
    main()

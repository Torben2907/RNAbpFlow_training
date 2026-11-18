"""
Compares the flow and the hybrid sampler on the held-out validation split.

Every validation RNA is sampled `--num-samples` times per sampler, with the same
seeds (and therefore the same initial noise) for all samplers, and each sample is
compared with the ground truth on the C1' atoms: RMSD, TM-score and lDDT, plus the
accuracy of the generated contact tokens.

    # one shard per GPU (run from anywhere; paths are relative to the project root)
    python3 scripts/compare_samplers.py run --ckpt <ckpt> --out results/compare \\
        --shard 0 --num-shards 4 --device cuda:0
    # after all shards are done
    python3 scripts/compare_samplers.py summarize --out results/compare

`run` writes one row per sample to <out>/samples_<shard>.csv (appended after every
RNA, so finished RNAs survive an interruption and are skipped on a rerun);
`summarize` combines the shards into <out>/summary.md.
"""

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, wilcoxon

from astrafold.config.hydra_schema import HybridSamplerConfig
from astrafold.data import base_constants as rc
from astrafold.data.pdb_na_dataset_base import PDBNABaseDataset
from astrafold.inference import Sampler
from astrafold.metrics import kabsch_rmsd, lddt, tm_score

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
    "self_consistency",
    "num_forced",
    "num_model_calls",
    "seconds",
]


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
    dataset = PDBNABaseDataset(data_cfg, is_training=False, csv_path=args.val_csv)
    indices = list(range(len(dataset)))[args.shard :: args.num_shards]
    if args.limit is not None:
        indices = indices[: args.limit]
    seeds = [args.seed + i for i in range(args.num_samples)]

    write_header = not rows_path.exists()
    with open(rows_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if write_header:
            writer.writeheader()
        for count, idx in enumerate(indices, start=1):
            pdb_name = dataset.csv.iloc[idx]["pdb_name"]
            example = {k: torch.as_tensor(v) for k, v in dataset[idx].items()}
            sequence = "".join(rc.restypes[a] for a in example["aatype"].tolist())
            mask = example["res_mask"].bool()
            gt = example["trans_1"]  # the C1' atoms
            gt_tokens = example["contact_tokens"].numpy()

            for name, sampler in samplers.items():
                if (pdb_name, name) in done:
                    continue
                start = time.time()
                result = sampler.sample(sequence, seeds)
                seconds = (time.time() - start) / len(seeds)

                pred = torch.from_numpy(result["atom_positions"][:, :, C1]).float()
                rmsd = kabsch_rmsd(
                    pred,
                    gt.expand(len(seeds), -1, -1),
                    mask.expand(len(seeds), -1),
                )
                for j, seed in enumerate(seeds):
                    p, g = pred[j][mask].numpy(), gt[mask].numpy()
                    tokens = result["contact_tokens"][j]
                    row = {
                        "pdb_name": pdb_name,
                        "length": len(sequence),
                        "sampler": name,
                        "seed": seed,
                        "rmsd": rmsd[j].item(),
                        "tm_score": tm_score(p, g),
                        "lddt": lddt(p, g),
                        "token_accuracy": (tokens == gt_tokens)[mask.numpy()].mean(),
                        "seconds": seconds,
                    }
                    for key in ("self_consistency", "num_forced", "num_model_calls"):
                        if key in result:
                            row[key] = result[key][j].item()
                    writer.writerow(row)
                f.flush()
            print(
                f"[shard {args.shard}] {count}/{len(indices)} {pdb_name} "
                f"(N={len(sequence)}) done",
                flush=True,
            )


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
        "# Flow vs hybrid sampler on the validation split",
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
    os.chdir(PROJECT_ROOT)  # the metadata CSVs hold paths relative to the root
    torch.set_float32_matmul_precision("high")
    {"run": run, "summarize": summarize}[args.command](args)


if __name__ == "__main__":
    main()

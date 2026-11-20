"""
Split a metadata CSV (by default metadata/RNA3DB_train.csv) into a training and a
held-out validation CSV.

RNA3DB (2024-04-26 release, the one used by RNAbpFlow) splits its chains into a train
and a test set only; its split.json has an empty valid_set. The validation set
therefore has to come out of the RNA3DB train set, and RNA3DB's own grouping does not
help with that:
  - its sequence clusters: the 560 chains of RNA3DB_train.csv are one representative
    each of 560 different clusters, so there is nothing left to group;
  - its components (the unit of its train/test split): one component holds 454 of the
    560 chains, so a component-level split would either move most of the data to
    validation or leave only a few small, unstratified families there.

So this script clusters the chains itself, so that near-identical RNAs never end up
on both sides of the split. Two chains are linked if
  - they come from the same PDB entry, or
  - their sequences share most of their k-mers (overlap coefficient
    |A & B| / min(|A|, |B|) >= --kmer-overlap, on the sets of k-mers).
Linked chains are merged into clusters (connected components), and whole clusters are
assigned to the validation set. With the defaults this gives 296 clusters, nested
within the RNA3DB components (99.4% of the linked pairs share a component). Most
validation chains thus share a component with training chains: validation measures
generalization to new sequences of known RNA families, while the RNA3DB test set holds
out whole components, so expect the validation loss to be optimistic relative to it.

The split is stratified by length: rows are binned into length quantiles and each bin
contributes about --val-fraction of its rows. Clusters larger than a bin's remaining
budget stay in training, so the big tRNA cluster is never moved to validation.

Usage:
    python scripts/make_val_split.py
    python scripts/make_val_split.py --csv metadata/RNA3DB_train.csv \
        --train-out metadata/RNA3DB_train_split.csv --val-out metadata/RNA3DB_val.csv
"""

import argparse

import numpy as np
import pandas as pd

from astrafold.data import utils as du
from astrafold.data.sequence import ONEHOT_ORDER


def read_sequence(processed_path: str) -> str:
    onehot = du.read_pkl(processed_path)["onehot"]
    return "".join(ONEHOT_ORDER[i] for i in np.asarray(onehot).argmax(-1))


def kmers(seq: str, k: int) -> set[str]:
    return {seq[i : i + k] for i in range(max(len(seq) - k + 1, 1))}


def cluster_chains(df: pd.DataFrame, k: int, min_overlap: float) -> np.ndarray:
    """Returns a cluster id per row (union-find over the links described above)."""
    n = len(df)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        parent[find(a)] = find(b)

    pdb_ids = df["pdb_name"].str[:4].str.lower().tolist()
    kmer_sets = [kmers(s, k) for s in df["sequence"]]

    for i in range(n):
        for j in range(i + 1, n):
            if pdb_ids[i] == pdb_ids[j]:
                union(i, j)
                continue
            overlap = len(kmer_sets[i] & kmer_sets[j]) / min(
                len(kmer_sets[i]), len(kmer_sets[j])
            )
            if overlap >= min_overlap:
                union(i, j)

    return np.array([find(i) for i in range(n)])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--csv", default="metadata/RNA3DB_train.csv")
    parser.add_argument("--train-out", default="metadata/RNA3DB_train_split.csv")
    parser.add_argument("--val-out", default="metadata/RNA3DB_val.csv")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--num-length-bins", type=int, default=5)
    parser.add_argument("--kmer-size", type=int, default=5)
    parser.add_argument("--kmer-overlap", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    original_columns = list(df.columns)
    df["sequence"] = [read_sequence(p) for p in df["processed_path"]]
    df["cluster"] = cluster_chains(df, args.kmer_size, args.kmer_overlap)
    df["length_bin"] = pd.qcut(
        df["modeled_na_seq_len"],
        q=args.num_length_bins,
        labels=False,
        duplicates="drop",
    )

    # A cluster belongs to the length bin of its median member.
    clusters = df.groupby("cluster").agg(
        size=("pdb_name", "size"), median_len=("modeled_na_seq_len", "median")
    )
    bin_edges = df.groupby("length_bin")["modeled_na_seq_len"].max().sort_index()
    clusters["length_bin"] = np.searchsorted(
        bin_edges.to_numpy(), clusters["median_len"].to_numpy()
    ).clip(max=len(bin_edges) - 1)

    rng = np.random.default_rng(args.seed)
    val_clusters: list[int] = []
    for length_bin, bin_clusters in clusters.groupby("length_bin"):
        budget = round(args.val_fraction * bin_clusters["size"].sum())
        taken = 0
        for cluster_id in rng.permutation(bin_clusters.index.to_numpy()):
            size = bin_clusters.loc[cluster_id, "size"]
            if taken + size <= budget:
                val_clusters.append(cluster_id)
                taken += size

    is_val = df["cluster"].isin(val_clusters)
    train_df = df.loc[~is_val, original_columns]
    val_df = df.loc[is_val, original_columns]

    # Sanity checks: no shared PDB entries or sequences across the split.
    assert not set(train_df["pdb_name"].str[:4]) & set(val_df["pdb_name"].str[:4])
    assert not set(df.loc[~is_val, "sequence"]) & set(df.loc[is_val, "sequence"])

    train_df.to_csv(args.train_out, index=False)
    val_df.to_csv(args.val_out, index=False)

    print(
        f"{len(df)} chains in {df['cluster'].nunique()} clusters "
        f"(largest: {clusters['size'].max()})"
    )
    print(f"train: {len(train_df)} -> {args.train_out}")
    print(f"val:   {len(val_df)} -> {args.val_out}")
    summary = (
        pd.DataFrame(
            {
                "train": df.loc[~is_val].groupby("length_bin").size(),
                "val": df.loc[is_val].groupby("length_bin").size(),
                "max_len": bin_edges,
            }
        )
        .fillna(0)
        .astype(int)
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# Downloads the native structures of the CASP15 RNA benchmark (the 10 targets of
# benchmarks/CASP15/CASP15_test.fasta, the test list of RNAbpFlow) from the CASP
# prediction center. The structures are not part of the repository.
#
#   scripts/download_casp15.sh [OUT_DIR]     (default: benchmarks/CASP15/natives)
#
# R1149 (10 models) and R1156 (4 conformations of 10 models) are extracted into
# subfolders; the other targets are single files, R1136 and R1138 with two
# conformations each.
set -euo pipefail

URL="https://predictioncenter.org/download_area/CASP15/targets/_4invitees/casp15_targets_RNA_4invitees.tgz"
SHA256="89d694dd02a833122582fa9711ff576f1fc4363d528490511c896b403cb0083b"

FILES=(
  R1107_D_1292119758_model-annotate_P1human.pdb
  R1108_D_1292119797_model-annotate_P1chimp.pdb
  R1116_PV_CL_Final.pdb
  R1117.pdb
  R1126.pdb
  R1128_structure_24_6wj_refined_046.pdb
  R1136v1.pdb
  R1136v2.pdb
  R1138v1_structure_23v1_6hbc_young.pdb
  R1138v2_structure_23v2_6hbc_mature.pdb
)
ENSEMBLES=(R1149 R1156)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/benchmarks/CASP15/natives}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "Downloading $URL"
curl -fsSL "$URL" -o "$TMP/rna.tgz"

if command -v sha256sum >/dev/null; then
  actual="$(sha256sum "$TMP/rna.tgz" | cut -d' ' -f1)"
else
  actual="$(shasum -a 256 "$TMP/rna.tgz" | cut -d' ' -f1)"
fi
if [[ "$actual" != "$SHA256" ]]; then
  echo "Checksum mismatch: expected $SHA256, got $actual" >&2
  echo "The archive on the prediction center has changed; check it before use." >&2
  exit 1
fi

members=("${FILES[@]}")
for t in "${ENSEMBLES[@]}"; do members+=("$t.tgz"); done
tar -xzf "$TMP/rna.tgz" -C "$TMP" "${members[@]}"

mkdir -p "$OUT"
for f in "${FILES[@]}"; do
  cp "$TMP/$f" "$OUT/"
done
for t in "${ENSEMBLES[@]}"; do
  mkdir -p "$OUT/$t"
  tar -xzf "$TMP/$t.tgz" -C "$OUT/$t"
done

echo "Wrote $(find "$OUT" -name '*.pdb' | wc -l | tr -d ' ') native structures to $OUT"

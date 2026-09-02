#!/usr/bin/env bash
set -euo pipefail
ulimit -n 65000

DATASET=""
GPU=""
CKPT=""
FINETUNE=false

TRAIN_EPOCHS=1500
FT_EPOCHS=200
FT_LR="0.00001"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)  DATASET="$2"; shift 2 ;;
    --gpu)      GPU="$2"; shift 2 ;;
    --ckpt)     CKPT="$2"; shift 2 ;;
    --finetune) FINETUNE=true; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

[[ -n "$DATASET" && -n "$GPU" ]] || {
  echo "Usage: $0 --dataset {RNA3DB|CASP16|CASP15} --gpu 0,1,2,3 [--finetune --ckpt PATH]"
  exit 1
}

# project name depends on mode
PROJ="train"
[[ "$FINETUNE" == "true" ]] && PROJ="Finetune"

# choose CSV based on dataset + mode
case "$DATASET" in
  RNA3DB)
    if [[ "$FINETUNE" == "true" ]]; then
      echo "RNA3DB finetune not supported in this script."
      exit 1
    fi
    CSV="metadata/RNA3DB_train.csv"
    NAME="RNA3DB"
    ;;
  CASP16)
    CSV="metadata/CASP16_train.csv"
    [[ "$FINETUNE" == "true" ]] && CSV="metadata/CASP16_FT.csv"
    NAME="CASP16"
    ;;
  CASP15)
    CSV="metadata/CASP15_train.csv"
    [[ "$FINETUNE" == "true" ]] && CSV="metadata/CASP15_FT.csv"
    NAME="CASP15"
    ;;
  *)
    echo "Usage: $0 --dataset {RNA3DB|CASP16|CASP15} --gpu 0,1,2,3 [--finetune --ckpt PATH]"
    exit 1
    ;;
esac

NUM_DEVICES=$(awk -F',' '{print NF}' <<< "$GPU")

MAX_EPOCHS="$TRAIN_EPOCHS"
[[ "$FINETUNE" == "true" ]] && MAX_EPOCHS="$FT_EPOCHS"

CMD=(python3 train.py
  data_cfg.csv_path="$CSV"
  experiment.wandb.project="$PROJ"
  experiment.wandb.name="$NAME"
  experiment.num_devices="$NUM_DEVICES"
  experiment.trainer.max_epochs="$MAX_EPOCHS"
)

if [[ "$FINETUNE" == "true" ]]; then
  [[ -n "$CKPT" ]] || { echo "--finetune requires --ckpt PATH"; exit 1; }
  CMD+=(experiment.finetune=true "experiment.warm_start=\"${CKPT}\"" experiment.optimizer.lr="$FT_LR")
else
  CMD+=(experiment.finetune=false experiment.warm_start=null)
fi

CUDA_VISIBLE_DEVICES="$GPU" "${CMD[@]}" > trlog.txt

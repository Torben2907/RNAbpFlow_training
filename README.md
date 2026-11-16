# RNAbpFlow training

Training code for *"Bridging discrete and continuous diffusion models for RNA structure
prediction"*.

The model is an SE(3) flow matching network for RNA backbones. It takes the noisy
nucleotide frames, the sequence and a partially masked set of per-nucleotide contact
tokens, and predicts the clean structure together with the masked tokens.

## Contents

- [Installation](#installation)
- [Data](#data)
- [Training](#training)
- [Validation](#validation)
- [Configuration](#configuration)
- [Tests](#tests)
- [Project layout](#project-layout)

## Installation

The dependencies are managed with [conda](https://docs.conda.io/projects/conda/en/latest/index.html).

```shell
conda env create -f RNAbpFlow.yml   # create the environment
conda activate RNAbpFlow
pip install -e .                    # install the rnabpflow package itself
```

Logging goes to [Weights & Biases](https://wandb.ai). Run `wandb login` once before the
first training, or add `experiment.wandb.mode=offline` to log locally.

## Data

The data is not part of the repository. Training expects three directories in the
project root:

| Path | Content |
| --- | --- |
| `metadata/<dataset>.csv` | One row per RNA chain: name, length and the path to its feature file |
| `Feature/train_pkl/<name>/<name>.pkl` | Masks and one-hot sequence of the chain |
| `pkls/<name>_base/<name>_feat.pkl` | Backbone frames and torsion angles of the chain |

Chains are filtered by length (`data_cfg.filtering`, 30 to 800 nucleotides by default).

### Train/validation split

`metadata/RNA3DB_train.csv` is split once into a training and a held-out validation set:

```shell
python3 scripts/make_val_split.py
# -> metadata/RNA3DB_train_split.csv (90%), metadata/RNA3DB_val.csv (10%)
```

Chains from the same PDB entry, or with near-identical sequences (5-mer overlap of at
least 0.7), always end up on the same side of the split. The split is stratified by
length and reproducible (`--seed`). See `--help` for all options.

## Training

```shell
python3 train.py run=train_full                                       # GPUs 0-3, from the preset
python3 train.py run=train_full experiment.trainer.devices='[4,5,6,7]' # exact GPU ids
python3 train.py run=train_full experiment.trainer.devices=2          # a count: the first 2 GPUs
python3 train.py run=smoke_overfit experiment.trainer.devices='[2]'
```

| Preset | Purpose |
| --- | --- |
| `run=train_full` | Full training on the RNA3DB split, with validation after every epoch |
| `run=smoke_overfit` | Debugging: overfits a single, fixed batch; no validation |

Without a preset, the defaults in `configs/` apply.

Checkpoints and the resolved config are written to
`ckpt/<wandb project>/<wandb name>/`. `train_full` keeps the 10 checkpoints with the
lowest validation loss, plus the last one.

To continue from a checkpoint:

```shell
# Resume a run (weights, optimizer state and step count)
python3 train.py run=train_full experiment.warm_start=path/to/last.ckpt

# Finetune: load only the weights and start a new run
python3 train.py run=train_full experiment.finetune=true experiment.warm_start=path/to/model.ckpt
```

`run.sh` wraps these commands for the RNA3DB, CASP15 and CASP16 datasets, including
finetuning:

```shell
./run.sh --dataset CASP16 --gpu 0,1,2,3
./run.sh --dataset CASP16 --gpu 0,1,2,3 --finetune --ckpt path/to/model.ckpt
```

> **Epochs are not passes over the data.** One epoch is `ceil(num_train_rnas / num_gpus)`
> steps per GPU, whatever the batch size. `train_full` therefore sets its length in
> optimizer steps (`experiment.trainer.max_steps`).

## Validation

Validation runs when `data_cfg.val_csv_path` is set, as in `train_full`. After every
epoch, each held-out RNA is evaluated at a fixed grid of flow times
(`experiment.validation.t_values`) with fixed noise and token masks, so the metrics
only change when the model does.

| Metric | Meaning |
| --- | --- |
| `valid/loss` | The training loss on the held-out set, averaged over the flow times |
| `valid/x1_rmsd` | RMSD (Å) between the one-step structure prediction and the ground truth, after superposition |
| `valid/loss_t<t>`, `valid/x1_rmsd_t<t>` | The same two metrics at a single flow time |
| `valid/<name>` | The individual loss terms and the token accuracy, as in training |

Set `data_cfg.val_csv_path=null` to train without validation. The checkpoint monitor
must then be a training metric (`experiment.checkpointer.monitor=loss`).

## Configuration

The config is composed by [Hydra](https://hydra.cc) from the groups in `configs/`.
`configs/config.yaml` selects one file per group, and a preset from `configs/run/`
overrides them.

| Group | Configures |
| --- | --- |
| `data/` | Metadata CSVs, length filter, batch sizes, contact tokens |
| `interpolant/` | Noise schedules of the flow |
| `model/` | Network architecture |
| `training/` | Loss weights and token masking |
| `validation/` | Flow times and seed of the validation step |
| `trainer/` | PyTorch Lightning trainer: devices, run length, logging interval |
| `checkpointer/` | Which checkpoints are kept |
| `wandb/` | Project and run name |
| `run/` | Presets |

Any value can be overridden on the command line, e.g. `experiment.optimizer.lr=0.0001`,
and a whole group can be swapped with `model=<file name without .yaml>`.
`src/rnabpflow/config/hydra_schema.py` holds the typed schema of the config.

## Tests

```shell
python3 -m pytest
```

The tests need the data and `pytest`. The validation tests are skipped until the
split has been created.

## Project layout

```
train.py                  training entry point
run.sh                    wrapper for the RNA3DB / CASP15 / CASP16 runs
configs/                  Hydra config groups and presets
scripts/make_val_split.py creates the train/validation split
src/rnabpflow/
  config/                 typed config schema
  data/                   dataset, batching, interpolant (noising), contact tokens
  models/                 network (flow_model.py) and Lightning module (flow_module.py)
tests/
```

## Acknowledgements

Large parts of the code are adapted from
[protein-frame-flow](https://github.com/microsoft/protein-frame-flow),
[MMDiff](https://github.com/Profluent-Internships/MMDiff) and
[OpenFold](https://github.com/aqlaboratory/openfold).

## License

MIT

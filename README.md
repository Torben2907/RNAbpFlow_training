# AstraFold

Training and inference code for *"Bridging discrete and continuous diffusion models for
RNA structure prediction"*.

The model is an SE(3) flow matching network for RNA backbones. It takes the noisy
nucleotide frames, the sequence and a partially masked set of per-nucleotide contact
tokens, and predicts the clean structure together with the masked tokens.

The repository covers both sides:

- **Training** (`train.py`): learns the model from experimental RNA structures and
  validates it on a held-out split.
- **Inference** (`inference.py`): samples all-atom structures, written as PDB or mmCIF,
  for RNA sequences with a trained checkpoint. It needs only the sequence, not the
  training data.

## Quick start

```shell
# Train (needs the data, see below)
python3 train.py run=train_full

# Predict structures for the sequences of a FASTA file
python3 inference.py ckpt_path=path/to/model.ckpt fasta_path=targets.fasta
```

## Contents

- [Quick start](#quick-start)
- [Installation](#installation)
- [Data](#data)
- [Training](#training)
- [Validation](#validation)
- [Inference](#inference)
- [Configuration](#configuration)
- [Tests](#tests)
- [Project layout](#project-layout)

## Installation

The dependencies are managed with [conda](https://docs.conda.io/projects/conda/en/latest/index.html).

```shell
conda env create -f requirements.yaml   # create the environment (pinned versions)
conda activate AstraFold
pip install -e .                        # install the astrafold package itself
```

`requirements.yaml` pins the exact versions of the environment used for training. The
same environment is used for inference.

Training logs go to [Weights & Biases](https://wandb.ai). Run `wandb login` once before
the first training, or add `experiment.wandb.mode=offline` to log locally. Inference
does not log to Weights & Biases.

## Data

This section is only needed for training; inference takes its input from a FASTA file.

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

## Inference

`inference.py` samples all-atom structures for the sequences of a FASTA file, with a
checkpoint written by `train.py`:

```shell
python3 inference.py ckpt_path=ckpt/AstraFold/full_4gpu_val/last.ckpt fasta_path=targets.fasta
python3 inference.py ckpt_path=... fasta_path=... num_samples=20 output_format=both device=cuda:1
```

The only input is the sequence (A, C, G, U; T is read as U). The structures are written
to `predictions/<FASTA record id>/Sample_<i>.pdb`. A FASTA file may hold any number of
sequences:

```
>hairpin
GGACUUCGGUCC
>my_target
GGCGCUUAGCGUAGAGCGCC
```

Inference runs on a GPU if one is available and also works on the CPU, only slower.

| Option | Default | Meaning |
| --- | --- | --- |
| `ckpt_path`, `fasta_path` | required | Checkpoint and input sequences |
| `output_dir` | `predictions` | Where the structures are written |
| `num_samples` | 10 | Structures per sequence |
| `batch_size` | 10 | Samples generated in one forward pass |
| `num_timesteps` | from the checkpoint (50) | Steps of the sampler |
| `seed` | 42 | Sample `i` is generated from `seed + i`, whatever the batch size |
| `output_format` | `pdb` | `pdb`, `mmcif` or `both` |
| `device` | `auto` | `auto` (GPU if available), `cpu`, `cuda` or `cuda:<id>` |

The sampler starts from random frames and fully masked contact tokens. It integrates
the frames from noise to data with Euler steps; in every step the model also predicts
the contact tokens, and a share of the masked ones is revealed, so that all tokens are
known when the flow reaches the data.

The same is available from Python:

```python
from astrafold.inference import Sampler

sampler = Sampler("path/to/model.ckpt")
result = sampler.sample("GGACUUCGGUCC", seeds=[0, 1])
result["atom_positions"]  # [2, 12, 28, 3], in Angstrom
```

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

Inference has its own config, `configs/inference.yaml` (see [Inference](#inference)).

Any value can be overridden on the command line, e.g. `experiment.optimizer.lr=0.0001`,
and a whole group can be swapped with `model=<file name without .yaml>`.
`astrafold/config/hydra_schema.py` holds the typed schema of the config.

## Tests

```shell
python3 -m pytest
```

The tests need the data and `pytest`, and run on the CPU. They cover training,
validation and inference. The validation tests are skipped until the split has been
created.

## Project layout

```
train.py                  training entry point
inference.py              inference entry point
run.sh                    wrapper for the RNA3DB / CASP15 / CASP16 runs
configs/                  Hydra config groups and presets
scripts/make_val_split.py creates the train/validation split
astrafold/                 the Python package
  config/                 typed config schema
  data/                   dataset, batching, interpolant (noising and sampling),
                          contact tokens, sequence features, PDB/mmCIF output
  models/                 network (flow_model.py) and Lightning module (flow_module.py)
  inference.py            sampler that turns a checkpoint and a sequence into structures
tests/
```

## Acknowledgements

Large parts of the code are adapted from
[protein-frame-flow](https://github.com/microsoft/protein-frame-flow),
[MMDiff](https://github.com/Profluent-Internships/MMDiff) and
[OpenFold](https://github.com/aqlaboratory/openfold). The sampler and the structure
output are adapted from the
[RNAbpFlow inference code](https://github.com/Torben2907/RNAbpFlow_inference), the PDB
writer from [NuFold](https://github.com/kiharalab/NuFold).

## License

MIT

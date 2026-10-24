# Training Code for "Bridging discrete and continuous diffusion for RNA structure prediction"

## Install
Use [conda](https://docs.conda.io/projects/conda/en/latest/index.html) to create a virtual environment and install all dependencies of RNAbpFlow.

```shell
# 0. Install conda (if you don't already have it)
curl -O https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
# 1. Install the conda environment from the .yaml-file
conda install --file RNAbpFlow.yaml
# 2. Activate the virtual environment
conda activate RNAbpFlow
```

## Run a training
You can fire up a **training** run via the commands:

```shell
python3 train.py run=smoke_overfit # GPUs 0,3 (from the preset)
python3 train.py run=smoke_overfit experiment.trainer.devices='[2]' # only GPU 2
python3 train.py run=train_full experiment.trainer.devices='[4,5,6,7]'
python3 train.py experiment.trainer.devices=4  # a count: the first 4 GPUs
```

The config is split into groups under `configs/` (`data/`, `interpolant/`, `model/`, `training/`,
`trainer/`, `wandb/`, `checkpointer/`) which `configs/config.yaml` composes. Presets in `configs/run/`
override them; a group can also be swapped from the CLI, e.g. `model=<file name without .yaml>`.

Information:
- `smoke_overfit` fill trigger a test run using only one batch of the same RNA data (used for debugging).
- `full_training` starts a full on training on `RNA3DB` by default, modify the path to the 
.csv-file for training on a different dataset.
- The training uses [WandB](https://forge.coreweave.com/wandb) for logging, be sure to use 

```shell
wandb login
```

for authentication when using for the fist time.

## License
MIT
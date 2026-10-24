# Training Code for "Bridging discrete and continuous diffusion for RNA structure prediction"

## Install
Use [conda](https://docs.conda.io/projects/conda/en/latest/index.html) to create a virtual environment and install all dependencies of RNAbpFlow.

```python
# 0. Install conda (if you don't already have it)
curl -O https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
# 1. Install the conda environment from the .yaml-file
conda install --file RNAbpFlow.yaml
# 2. Activate the virtual environment
conda activate RNAbpFlow
```

## Run a training
You can fire up a **training** run via the commands:

```python
python3 train.py --config-name=smoke_overfit # GPUs 0,1,2,3 (from the preset)
python3 train.py --config-name=smoke_overfit experiment.trainer.devices='[2]' # only GPU 2
python3 train.py --config-name=train_full experiment.trainer.devices='[4,5,6,7]'
python3 train.py experiment.trainer.devices=4  # a count: the first 4 GPUs
```

Information:
- `smoke_overfit` fill trigger a test run using only one batch of the same RNA data (used for debugging).
- `full_training` starts a full on training on `RNA3DB` by default, modify the path to the 
.csv-file for training on a different dataset.

## License
MIT
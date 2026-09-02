Train RNAbpFlow:
=======================================================
1. chmod +x run.sh

2. Use mamba to create a virtual environment and install dependencies for RNAbpFlow

conda install -n base -c conda-forge mamba
mamba env create -f RNAbpFlow.yml

3. Activate the virtual environment
conda activate RNAbpFlow

Training
================================================================

1. For the provided RNA3DB train-test split experiemnt

./run.sh --dataset RNA3DB --gpu 0,1,2,3,4,5,6,7

2. For the blind CASP15 experiemnt

./run.sh --dataset CASP15 --gpu 0,1,2,3,4,5,6,7

3. For the blind CASP16 experiemnt

./run.sh --dataset CASP16 --gpu 0,1,2,3,4,5,6,7


Finetune an already trained model based on predicted base pairs:
================================================================

Commands:

For CASP16:

./run.sh --dataset CASP16 --gpu 0,1,2,3,4,5,6,7 --finetune --ckpt /path/to/model.ckpt

For CASP15:

./run.sh --dataset CASP15 --gpu 0,1,2,3,4,5,6,7 --finetune --ckpt /path/to/model.ckpt

** All the trained/finetunned checkpoints can be found in ckpt/ folder.

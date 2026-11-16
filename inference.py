"""
Inference entry point: samples RNA structures for the sequences of a FASTA file.

Usage: python3 inference.py ckpt_path=<checkpoint> fasta_path=<FASTA file>
(all options: configs/inference.yaml)
"""

from typing import cast

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from astrafold.config.hydra_schema import InferenceConfig
from astrafold.inference import run_inference

torch.set_float32_matmul_precision("high")


@hydra.main(version_base=None, config_path="./configs", config_name="inference")
def main(cfg: DictConfig):
    # Merging with the schema checks the option names and types.
    schema = OmegaConf.structured(InferenceConfig)
    run_inference(cast(InferenceConfig, OmegaConf.merge(schema, cfg)))


if __name__ == "__main__":
    main()

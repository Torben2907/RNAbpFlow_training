"""
Dataset of single-chain RNA structures, read from preprocessed feature pickles.

Code adapted from MMDiff (https://github.com/Profluent-Internships/MMDiff).
"""

import functools as fn
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import tree
from beartype.typing import Optional
from torch.utils.data import Dataset

from astrafold.config.hydra_schema import DataConfig
from astrafold.data import rigid_utils
from astrafold.data import utils as du
from astrafold.data.contacts import make_contact_tokens

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class PDBNABaseDataset(Dataset):
    """
    Each row of the metadata CSV is one RNA chain. Its features come from two pickles:
        <processed_path>                             : masks and the one-hot sequence
        pkls/<pdb_name>_base/<pdb_name>_feat.pkl     : frames and torsion angles

    Params:
        data_conf   : data config (length filters, contact token settings)
        is_training : only labels the log message
        csv_path    : metadata CSV to read; defaults to `data_conf.csv_path`
    """

    def __init__(
        self,
        data_conf: DataConfig,
        is_training: bool,
        csv_path: Optional[str] = None,
    ):
        self._data_conf = data_conf
        self._csv_path = data_conf.csv_path if csv_path is None else csv_path
        self._is_training = is_training
        self.csv = self._read_metadata()

    @property
    def is_training(self):
        return self._is_training

    def _read_metadata(self) -> pd.DataFrame:
        """Reads the metadata CSV and keeps the RNA-only monomers within the length limits."""
        pdb_csv = pd.read_csv(self._csv_path)

        filter_conf = self._data_conf.filtering
        pdb_csv = pdb_csv[pdb_csv.modeled_na_seq_len <= filter_conf.max_len]
        pdb_csv = pdb_csv[pdb_csv.modeled_na_seq_len >= filter_conf.min_len]
        pdb_csv = pdb_csv[pdb_csv.quaternary_category == "homomer"]  # ignore multimers
        pdb_csv = pdb_csv[pdb_csv.num_protein_chains == 0]  # remove proteins
        pdb_csv = pdb_csv.sort_values("modeled_na_seq_len", ascending=False)

        split = "Training" if self.is_training else "Validation"
        print(f"{split}: {len(pdb_csv)} filtered examples from {self._csv_path}")
        return pdb_csv

    # Cached because the same example can be drawn several times per epoch.
    @fn.lru_cache(maxsize=100)  # noqa: B019 (the dataset lives as long as the process)
    def _process_csv_row(self, processed_file_path: str) -> dict:
        processed_feats = du.read_pkl(processed_file_path)
        onehot = processed_feats["onehot"]

        pdb_name = processed_file_path.split("/")[-2]
        nufeat = du.read_pkl(f"{PROJECT_ROOT}/pkls/{pdb_name}_base/{pdb_name}_feat.pkl")

        processed_feats = du.parse_complex_feats(processed_feats)
        molecule_type = processed_feats["molecule_type_encoding"]
        is_na_residue_mask = (molecule_type[:, 1] == 1) | (molecule_type[:, 2] == 1)

        # Only take the modeled residues
        modeled_idx = processed_feats["modeled_idx"]
        modeled = slice(np.min(modeled_idx), np.max(modeled_idx) + 1)
        is_na_residue_mask = is_na_residue_mask[modeled]
        bb_mask = torch.tensor(processed_feats["bb_mask"][modeled])

        # The backbone frame is the first rigid group of each nucleotide.
        rigids_1 = rigid_utils.Rigid.from_tensor_4x4(nufeat["rigidgroups_gt_frames"])[
            :, 0
        ]
        trans_1 = rigids_1.get_trans()

        contact_tokens = make_contact_tokens(
            trans_1,
            bb_mask.bool(),
            distance_threshold=self._data_conf.contact_distance_threshold,
            min_sequence_separation=self._data_conf.contact_minimum_sequence_separation,
        )

        return {
            "aatype": nufeat["aatype"],
            "onehot": onehot,
            "res_mask": bb_mask.int(),
            "is_na_residue_mask": is_na_residue_mask,
            "rigid_mask": nufeat["backbone_rigid_mask"],
            "rigidgroups_gt_exists": nufeat["rigidgroups_gt_exists"],
            "trans_1": trans_1,  # [N, 3] frame origins (in Angstrom)
            "rotmats_1": rigids_1.get_rots().get_rot_mats(),  # [N, 3, 3]
            "torsion_angles_sin_cos": nufeat["torsion_angles_sin_cos"],
            "contact_tokens": contact_tokens,  # [N] 1 = has a contact, 0 = has none
        }

    @staticmethod
    def _float64_to_float32(feats: dict) -> dict:
        converted = {}
        for key, value in feats.items():
            if isinstance(value, np.ndarray) and value.dtype == np.float64:
                converted[key] = value.astype(np.float32)
            elif isinstance(value, torch.Tensor) and value.dtype == torch.float64:
                converted[key] = value.float()
            else:
                converted[key] = value
        return converted

    def __getitem__(self, idx):
        csv_row = self.csv.iloc[idx]
        final_feats = self._process_csv_row(csv_row["processed_path"])

        # Convert all features to tensors.
        final_feats = tree.map_structure(
            lambda x: x if torch.is_tensor(x) else torch.tensor(x), final_feats
        )
        final_feats = du.pad_feats(final_feats, csv_row["modeled_na_seq_len"])
        return self._float64_to_float32(final_feats)

    def __len__(self):
        return len(self.csv)

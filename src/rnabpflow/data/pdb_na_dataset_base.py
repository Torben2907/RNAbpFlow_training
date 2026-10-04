# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for MMDiff (https://github.com/Profluent-Internships/MMDiff):
# -------------------------------------------------------------------------------------------------------------------------------------

import functools as fn
import numpy as np
import pandas as pd

import torch, tree, os
from beartype.typing import Optional
from omegaconf import DictConfig
from torch.utils.data import Dataset

from rnabpflow.data import data_transforms
from rnabpflow.data import utils as du
from rnabpflow.data import rigid_utils

from rnabpflow.data.contacts import make_contact_tokens

NUM_NA_RESIDUE_ATOMS = 23

class PDBNABaseDataset(Dataset):
    def __init__(
        self,
        data_conf,
        is_training: bool,
        filter_eval_split: bool = False,
        inference_cfg: Optional[DictConfig] = None,
    ):
        self._data_conf = data_conf
        self._is_training = is_training
        self._filter_eval_split = filter_eval_split
        self._inference_cfg = inference_cfg
        self._init_metadata_and_splits()

    @property
    def data_conf(self):
        return self._data_conf

    @property
    def ddpm(self):
        return self._ddpm

    @property
    def is_training(self):
        return self._is_training

    @property
    def filter_eval_split(self):
        return self._filter_eval_split

    @property
    def inference_cfg(self):
        return self._inference_cfg

    @property
    def sequence_ddpm(self):
        return self._sequence_ddpm

    def _init_metadata_and_splits(self): # new
        pdb_csv = pd.read_csv(self.data_conf.csv_path)
        self.raw_csv = pdb_csv # original CSV before filtering and transformations

        pdb_csv.fillna(
            {"helix_percent": 0, "coil_percent": 0, "strand_percent": 0, "radius_gyration": 0},
            inplace=True,
        )

        filter_conf = self.data_conf.filtering

        pd.set_option('display.max_rows', None)
        result = pdb_csv[['pdb_name', 'modeled_na_seq_len']]
        #print(result)

        max_value = result['modeled_na_seq_len'].max()
        min_value = result['modeled_na_seq_len'].min()

        # Group by 'modeled_na_seq_len' and count the number of rows for each length
        length_counts = result.groupby('modeled_na_seq_len').size().reset_index(name='count')

        # Print the result
        #print(length_counts)

        # length-based filtering
        pdb_csv = pdb_csv[pdb_csv.modeled_na_seq_len <= filter_conf.max_len]
        pdb_csv = pdb_csv[pdb_csv.modeled_na_seq_len >= filter_conf.min_len]

        pdb_csv = pdb_csv[pdb_csv.quaternary_category == "homomer"] # ignore multimers
        pdb_csv = pdb_csv[pdb_csv.num_protein_chains == 0] # remove proteins
        pdb_csv = pdb_csv.sort_values("modeled_na_seq_len", ascending=False)

        #print(min_value,max_value)

        # pdb_csv.reset_index(inplace=True) # reset the index to ensure samples are taken within proper bounds

        if self.is_training:
            self.csv = pdb_csv
            print (f"Training: {len(self.csv)} filtered examples") # FINAL post-filtered set for RNASolo from metadata CSV
        else:
            # validation data
            eval_csv = pdb_csv
            all_lengths = pdb_csv['modeled_na_seq_len'].unique()
            length_indices = (len(all_lengths) - 1) * np.linspace(0.0, 1.0, self._data_conf.num_eval_lengths)
            length_indices = length_indices.astype(int)
            eval_lengths = all_lengths[length_indices]
            eval_csv = eval_csv[eval_csv.modeled_na_seq_len.isin(eval_lengths)]

            eval_csv = eval_csv.groupby("modeled_na_seq_len").sample(self._data_conf.samples_per_eval_length, replace=True, random_state=123)
            eval_csv = eval_csv.sort_values(["modeled_na_seq_len"], ascending=False)
            self.csv = eval_csv
            #print (f"Validation: {len(self.csv)} examples with lengths {eval_lengths}.")

    # cache make the same sample in same batch
    @fn.lru_cache(maxsize=100)
    def _process_csv_row(self, processed_file_path):
        processed_feats = du.read_pkl(processed_file_path)
        ss = processed_feats['ss'].astype(float)
        if "ss_pred" in processed_feats:
            ss_pred = processed_feats['ss_pred'].astype(float)
        oh = processed_feats['onehot']
        
        ssmask = processed_feats['custom_ss_mask'].astype(int)
        pdb_name = processed_file_path.split('/')[-2]

        current_directory = os.path.dirname(os.path.abspath(__file__))
        current_directory = os.path.dirname(current_directory)
        current_directory = os.path.dirname(current_directory)
        current_directory = os.path.dirname(current_directory)

        nufeat = du.read_pkl(f"{current_directory}/pkls/{pdb_name}_base/{pdb_name}_feat.pkl")
        
        processed_feats = du.parse_complex_feats(processed_feats)

        # Designate which residues to diffuse and which to fix. By default, diffuse all residues
        diffused_mask = np.ones_like(processed_feats["bb_mask"])
        if np.sum(diffused_mask) < 1:
            raise ValueError("Must be diffused")
        fixed_mask = 1 - diffused_mask
        processed_feats["fixed_mask"] = fixed_mask

        # Distinguish between protein residues and nucleic acid residues using corresponding masks
        processed_feats["is_protein_residue_mask"] = (
            processed_feats["molecule_type_encoding"][:, 0] == 1
        )
        processed_feats["is_na_residue_mask"] = (
            processed_feats["molecule_type_encoding"][:, 1] == 1
        ) | (processed_feats["molecule_type_encoding"][:, 2] == 1)
        na_inputs_present = processed_feats["is_na_residue_mask"].any().item()

        # Find interfaces
        inter_chain_interacting_residue_mask = torch.zeros(len(diffused_mask), dtype=torch.bool)
        inter_chain_interacting_residue_mask[processed_feats["inter_chain_interacting_idx"]] = True

        # Only take modeled residues
        modeled_idx = processed_feats["modeled_idx"]
        min_idx = np.min(modeled_idx)
        max_idx = np.max(modeled_idx)
        
        del processed_feats["modeled_idx"]
       
        if processed_feats["protein_modeled_idx"] is None:
            del processed_feats["protein_modeled_idx"]
       
        if processed_feats["na_modeled_idx"] is None:
            del processed_feats["na_modeled_idx"]
       
        processed_feats = tree.map_structure(lambda x: x[min_idx : (max_idx + 1)], processed_feats)
        inter_chain_interacting_residue_mask = inter_chain_interacting_residue_mask[
            min_idx : (max_idx + 1)
        ]

        # Run through OpenFold data transforms.
        chain_feats, na_chain_feats = (
            {
                "aatype": torch.tensor(processed_feats["aatype"]).long(),
                "all_atom_positions": torch.tensor(processed_feats["atom_positions"]).double(),
                "all_atom_mask": torch.tensor(processed_feats["atom_mask"]).double(),
                "atom_deoxy": torch.tensor(processed_feats["atom_deoxy"]).bool(),
            },
            {},
        )

        if na_inputs_present:
            na_chain_feats = {
                "aatype": chain_feats["aatype"][processed_feats["is_na_residue_mask"]],
                "all_atom_positions": chain_feats["all_atom_positions"][
                    processed_feats["is_na_residue_mask"]
                ][:, :NUM_NA_RESIDUE_ATOMS],
                "all_atom_mask": chain_feats["all_atom_mask"][
                    processed_feats["is_na_residue_mask"]
                ][:, :NUM_NA_RESIDUE_ATOMS],
                "atom_deoxy": chain_feats["atom_deoxy"][processed_feats["is_na_residue_mask"]],
            }
            na_chain_feats["atom23_gt_positions"] = na_chain_feats[
                "all_atom_positions"
            ]  # cache `atom23` positions
        
        if na_inputs_present:
            na_chain_feats = data_transforms.make_atom23_masks(na_chain_feats)
            data_transforms.atom23_list_to_atom27_list(
                na_chain_feats, ["all_atom_positions", "all_atom_mask"], inplace=True
            )
            na_chain_feats = data_transforms.atom27_to_frames(na_chain_feats)
            na_chain_feats = data_transforms.atom27_to_torsion_angles()(na_chain_feats)

        # Merge available protein and nucleic acid features using padding where necessary
        chain_feats = du.concat_complex_torch_features(
            chain_feats,
            {}, # empty protein features
            na_chain_feats,
            feature_concat_map=du.COMPLEX_FEATURE_CONCAT_MAP,
            add_batch_dim=False,
        )

        final_feats = {
            "torsion_angles_sin_cos": chain_feats["torsion_angles_sin_cos"],
            "is_na_residue_mask": processed_feats["is_na_residue_mask"],
            "ss": ss,
            "onehot": oh,
            "ssmask": ssmask,
            "aatype": nufeat["aatype"],
            "rigid_mask": nufeat["backbone_rigid_mask"],
            "rigidgroups_gt_exists": nufeat["rigidgroups_gt_exists"],
        }

        if "ss_pred" in processed_feats:
            final_feats["ss_pred"] = ss_pred
        
        rigids_1 = rigid_utils.Rigid.from_tensor_4x4(
                                         nufeat["rigidgroups_gt_frames"]
                                     )[:, 0]
        
        rotmats_1 = rigids_1.get_rots().get_rot_mats()
        trans_1 = rigids_1.get_trans()

        contact_tokens = make_contact_tokens(
            trans_1, 
            torch.tensor(processed_feats["bb_mask"]).bool(),
            distance_threshold=self._data_conf.contact_distance_threshold,
            min_sequence_separation=self._data_conf.contact_minimum_sequence_separation
        )

        final_feats["rotmats_1"] = rotmats_1
        final_feats["trans_1"] = trans_1
        final_feats["torsion_angles_sin_cos"] = nufeat["torsion_angles_sin_cos"]

        final_feats['res_mask'] = torch.tensor(processed_feats['bb_mask']).int()

        final_feats["contact_tokens"] = contact_tokens

        """
        Final sample dict keys:
            - torsion_angles_sin_cos
            - rotmats_1
            - trans_1
            - res_mask
            - is_na_residue_mask
        """

        # for k,v in final_feats.items():
        #    print(k, v.shape)

        return final_feats

    def convert_dict_float64_items_to_float32(self, dictionary):
        converted_dict = {}
        for key, value in dictionary.items():
            if isinstance(value, np.ndarray) and value.dtype == np.float64:
                converted_dict[key] = value.astype(np.float32)
            elif isinstance(value, torch.Tensor) and value.dtype == torch.float64:
                converted_dict[key] = value.float()
            else:
                converted_dict[key] = value  # For non-NumPy array and non-PyTorch tensor types
        return converted_dict
    
    def __getitem__(self, idx):
        example_idx = idx
        csv_row = self.csv.iloc[example_idx]
        processed_file_path = csv_row["processed_path"]
        final_feats = self._process_csv_row(processed_file_path) # get the features for this instance

        # Convert all features to tensors.
        final_feats = tree.map_structure(
            lambda x: x if torch.is_tensor(x) else torch.tensor(x), final_feats
        )
        final_feats = du.pad_feats(final_feats, csv_row["modeled_na_seq_len"])
        final_feats = self.convert_dict_float64_items_to_float32(final_feats)
        
        return final_feats
    
    def __len__(self):
        return len(self.csv)

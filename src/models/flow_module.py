"""
Code adapted from
https://github.com/microsoft/protein-frame-flow/blob/main/models/flow_module.py
"""

import torch
import time
import os
import random
import numpy as np
import pandas as pd
import logging
from pytorch_lightning import LightningModule
from torch import nn

from src.models.flow_model import FlowModel
from src.models import utils as mu
from src.data.interpolant import Interpolant 
from src.data import utils as du
from src.data import all_atom as rna_all_atom
from src.data import so3_utils
import torch.nn.functional as F
from src.data.rigid_utils import Rigid,Rotation

torch.autograd.set_detect_anomaly(True)

class FlowModule(LightningModule):
    def __init__(self, cfg, folding_cfg=None):
        super().__init__()
        self._print_logger = logging.getLogger(__name__)
        self._exp_cfg = cfg.experiment
        self._model_cfg = cfg.model
        self._interpolant_cfg = cfg.interpolant

        # Set-up vector field prediction model
        self.model = FlowModel(cfg.model)

        # Set-up interpolant
        self.interpolant = Interpolant(cfg.interpolant)

        self._sample_write_dir = self._exp_cfg.checkpointer.dirpath
        os.makedirs(self._sample_write_dir, exist_ok=True)

        self.validation_epoch_metrics = []
        self.validation_epoch_samples = []
        self.save_hyperparameters()
        
    def on_train_start(self):
        
        self._epoch_start_time = time.time()
        
    def on_train_epoch_end(self):
        
        trainer = self.trainer

    def model_step(self, noisy_batch):
        """
        Params:
            noisy_batch (dict) : dictionary of tensors corresponding to corrupted Frame objects

        Remarks:
            Computes the different core and auxiliary losses between ground truth and predicted backbones

        Returns:
            Dictionary of core and auxiliary losses
        """
        training_cfg = self._exp_cfg.training
        loss_mask = noisy_batch['res_mask']
        is_na_residue_mask = noisy_batch["is_na_residue_mask"]
        
        if training_cfg.min_plddt_mask is not None:
            plddt_mask = noisy_batch['res_plddt'] > training_cfg.min_plddt_mask
            loss_mask *= plddt_mask
        
        num_batch, num_res = loss_mask.shape

        torsions_start_index = 0
        torsions_end_index = 9
        num_torsions = torsions_end_index - torsions_start_index

        C1index= [11]

        # Ground truth labels
        gt_trans_1 = noisy_batch['trans_1']
        gt_rotmats_1 = noisy_batch['rotmats_1']
        gt_torsions_1 = noisy_batch['torsion_angles_sin_cos'][:, :, torsions_start_index:torsions_end_index, :].reshape(num_batch, num_res, num_torsions * 2)
        rotmats_t = noisy_batch['rotmats_t']
        gt_rot_vf = so3_utils.calc_rot_vf(rotmats_t, gt_rotmats_1.type(torch.float32))

        gt_bb_atoms_all = rna_all_atom.to_atom23_rna(
                            gt_trans_1, gt_rotmats_1, 
                            noisy_batch['aatype'],
                            torsions=noisy_batch['torsion_angles_sin_cos']
                        )

        gt_bb_atoms2 = gt_bb_atoms_all[:, :, C1index]

        # Timestep used for normalization.
        t = noisy_batch['t']
        norm_scale = 1 - torch.min(t[..., None], torch.tensor(training_cfg.t_normalize_clip))
        
        # Model output predictions.
        model_output = self.model(noisy_batch)
        pred_trans_1 = model_output['pred_trans']
        pred_rotmats_1 = model_output['pred_rotmats']
        pred_torsions_1 = model_output['pred_torsions'].reshape(num_batch, num_res, num_torsions * 2)
        pred_rots_vf = so3_utils.calc_rot_vf(rotmats_t, pred_rotmats_1)

        pred_bb_atoms_all = rna_all_atom.to_atom23_rna(
                            pred_trans_1, pred_rotmats_1, 
                            noisy_batch['aatype'],
                            torsions=model_output['pred_torsions']
                        )

        pred_bb_atoms2 = pred_bb_atoms_all[:, :, C1index]

        custom_pair_mask = noisy_batch['ssmask']

        # Translation VF loss
        trans_error = (gt_trans_1 - pred_trans_1) / norm_scale * training_cfg.trans_scale
        loss_denom = torch.sum(loss_mask, dim=-1) * 3  # 3 frame atoms
        trans_loss = training_cfg.translation_loss_weight * torch.sum(
            trans_error ** 2 * loss_mask[..., None],
            dim=(-1, -2)
        ) / loss_denom

        # Rotation VF loss
        rots_vf_error = (gt_rot_vf - pred_rots_vf) / norm_scale
        loss_denom = torch.sum(loss_mask, dim=-1) * 3  # 3 frame atoms
        rots_vf_loss = training_cfg.rotation_loss_weights * torch.sum(
            rots_vf_error ** 2 * loss_mask[..., None],
            dim=(-1, -2)
        ) / loss_denom

        gt_flat_atoms = gt_bb_atoms2.reshape([num_batch, num_res, 3]) 
        gt_pair_dists = torch.linalg.norm(gt_flat_atoms[:, :, None, :] - gt_flat_atoms[:, None, :, :], dim=-1)
        pred_flat_atoms = pred_bb_atoms2.reshape([num_batch, num_res, 3]) 
        pred_pair_dists = torch.linalg.norm(pred_flat_atoms[:, :, None, :] - pred_flat_atoms[:, None, :, :], dim=-1)

        flat_loss_mask = torch.tile(loss_mask[:, :, None], (1, 1, 1))
        flat_loss_mask = flat_loss_mask.reshape([num_batch, num_res])
        flat_res_mask = torch.tile(loss_mask[:, :, None], (1, 1, 1))
        flat_res_mask = flat_res_mask.reshape([num_batch, num_res]) 

        gt_pair_dists = gt_pair_dists * flat_loss_mask[..., None]
        pred_pair_dists = pred_pair_dists * flat_loss_mask[..., None]
        
        pair_dist = (gt_pair_dists - pred_pair_dists)**2

        pair_dist_mask1 = (flat_loss_mask[..., None] * flat_res_mask[:, None, :]) * custom_pair_mask[:,:,:,0]
        pair_dist_mask2 = (flat_loss_mask[..., None] * flat_res_mask[:, None, :]) * custom_pair_mask[:,:,:,1]
        pair_dist_mask3 = (flat_loss_mask[..., None] * flat_res_mask[:, None, :]) * custom_pair_mask[:,:,:,2]

        ss3D_loss_1 = torch.sum(pair_dist * pair_dist_mask1, dim=(1, 2))
        ss3D_loss_2 = torch.sum(pair_dist * pair_dist_mask2, dim=(1, 2))
        ss3D_loss_3 = torch.sum(pair_dist * pair_dist_mask3, dim=(1, 2))

        ss3D_loss = (ss3D_loss_1 + ss3D_loss_2 + ss3D_loss_3) / 3
        ss3D_loss /= ((torch.sum(pair_dist_mask1, dim=(1, 2))) + (torch.sum(pair_dist_mask2, dim=(1, 2))) + (torch.sum(pair_dist_mask3, dim=(1, 2))))

        predicted_logits = model_output['pair_feat']
        gtss = noisy_batch['ss']
        bceloss = nn.BCEWithLogitsLoss(reduction='none')

        ss2dloss = bceloss(predicted_logits, gtss) 
        ss2dloss = ss2dloss.mean(dim=(1, 2)) 
        ss2dloss = ss2dloss.sum(dim=-1)  

        # Torsion angles loss
        pred_torsions_1 = pred_torsions_1.reshape(num_batch, num_res, num_torsions, 2)
        gt_torsions_1 = gt_torsions_1.reshape(num_batch, num_res, num_torsions, 2)
        loss_denom = torch.sum(loss_mask, dim=-1) * num_torsions  # 9 torsion angles
        tors_loss = training_cfg.tors_loss_scale * torch.sum(
            torch.linalg.norm(pred_torsions_1 - gt_torsions_1, dim=-1) ** 2 * loss_mask[..., None], dim=(-1, -2)
        ) / loss_denom
        
        se3_vf_loss = trans_loss + rots_vf_loss
        auxiliary_loss = (ss3D_loss + ss2dloss + tors_loss) * (t[:, 0] > training_cfg.aux_loss_t_pass)
        auxiliary_loss *= self._exp_cfg.training.aux_loss_weight

        if torch.isnan(auxiliary_loss).any():
            print ("NaN loss in aux_loss")
            auxiliary_loss = torch.zeros_like(auxiliary_loss).to(se3_vf_loss.device)
        
        if torch.isnan(se3_vf_loss).any():
            # raise ValueError('NaN loss encountered')
            print ("NaN loss in se3_vf_loss")
            se3_vf_loss = torch.zeros_like(se3_vf_loss).to(se3_vf_loss.device)
        
        return {
            "trans_loss": trans_loss,
            "ss3D_loss": ss3D_loss,
            "ss2D_loss": ss2dloss,
            "auxiliary_loss": auxiliary_loss,
            "rots_vf_loss": rots_vf_loss,
            "se3_vf_loss": se3_vf_loss,
            "torsion_loss": tors_loss
        }
        
    def validation_step(self, batch, batch_idx):
        """
        Generates samples and computes (average) local structural metrics 
        such as C4'-C4' distances, steric clashes, gyration radius.
        """

        res_mask = batch['res_mask']
        is_na_residue_mask = batch['is_na_residue_mask'].bool()
        self.interpolant.set_device(res_mask.device)
        num_batch = res_mask.shape[0]
        num_res = is_na_residue_mask.sum(dim=-1).max().item()

        batch_metrics = []

        batch_metrics = pd.DataFrame(batch_metrics)
        self.validation_epoch_metrics.append(batch_metrics)
        
    def on_validation_epoch_end(self):
        if len(self.validation_epoch_samples) > 0:
            self.logger.log_table(
                key='valid/samples',
                columns=["sample_path", "global_step", "RNA"],
                data=self.validation_epoch_samples)
            self.validation_epoch_samples.clear()

        def extract_scalar(val):
            if isinstance(val, dict):
                if len(val) == 1:
                    return list(val.values())[0]  # Return the first value in the dict
                else:
                    raise ValueError("Dictionary has more than one value; cannot log.")
            elif isinstance(val, np.ndarray):
                if val.size == 1:
                    return val.item()  # Convert single-element array to scalar
                else:
                    raise ValueError("Array has more than one value; cannot log.")
            elif isinstance(val, list):
                if len(val) == 1:
                    return val[0]  # Return the first item if list has one element
                else:
                    raise ValueError("List has more than one value; cannot log.")
            elif isinstance(val, (int, float)):
                return val  # Directly return scalar values
            else:
                raise ValueError(f"Unsupported type for metric value: {type(val)}")

        # Assuming val_epoch_metrics is a DataFrame or similar structure
        val_epoch_metrics = pd.concat(self.validation_epoch_metrics)

        # Apply extract_scalar to all values before logging
        processed_metrics = {k: extract_scalar(v) for k, v in val_epoch_metrics.to_dict().items()}

        for metric_name, metric_val in processed_metrics.items():
            scalar_val = extract_scalar(metric_val)
            #print(f"Metric Name: {metric_name}, Metric Value: {metric_val}, Scalar Value: {scalar_val}, Type: {type(scalar_val)}")
            if isinstance(scalar_val, (int, float)):  # Check if the scalar value is valid
                self._log_scalar(
                    f'valid/{metric_name}',
                    scalar_val,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    batch_size=len(val_epoch_metrics),
                )
            else:
                raise ValueError(f"Metric value for {metric_name} is not a scalar: {scalar_val}")

        
        self.validation_epoch_metrics.clear()

    def _log_scalar(
            self,
            key,
            value,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=None,
            sync_dist=False,
            rank_zero_only=True
        ):
        if sync_dist and rank_zero_only:
            raise ValueError('Unable to sync dist when rank_zero_only=True')
        self.log(
            key,
            value,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=prog_bar,
            batch_size=batch_size,
            sync_dist=sync_dist,
            rank_zero_only=rank_zero_only
        )

    def training_step(self, batch, stage):
        """
        Performs one iteration of SE(3) flow matching and returns total training loss
        using the core and auxiliary losses computed in `model_step`.
        """
        step_start_time = time.time()
        self.interpolant.set_device(batch['res_mask'].device)
        noisy_batch = self.interpolant.corrupt_batch(batch)

        if self._interpolant_cfg.self_condition and random.random() > 0.5:
            with torch.no_grad():
                model_sc = self.model(noisy_batch)
                noisy_batch['trans_sc'] = model_sc['pred_trans']
        
        batch_losses = self.model_step(noisy_batch)
        num_batch = batch_losses['torsion_loss'].shape[0]
        
        total_losses = {
            k: torch.mean(v) for k,v in batch_losses.items()
        }

        for k,v in total_losses.items():
            self._log_scalar(f"train/{k}", v, prog_bar=False, batch_size=num_batch)
        
        # Losses to track. Stratified across t.
        t = torch.squeeze(noisy_batch['t'])
        self._log_scalar(
            "train/t",
            np.mean(du.to_numpy(t)),
            prog_bar=False, batch_size=num_batch)
        
        step_time = time.time() - step_start_time
        #self._log_scalar("train/eps", num_batch / step_time)
        
        train_loss = (
            total_losses[self._exp_cfg.training.loss] +
            total_losses['auxiliary_loss']
        )
        self._log_scalar("loss", train_loss, batch_size=num_batch)

        return train_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            params=self.model.parameters(),
            **self._exp_cfg.optimizer
        )

  
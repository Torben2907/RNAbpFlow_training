"""
Lightning module for training the flow model.

Code adapted from
https://github.com/microsoft/protein-frame-flow/blob/main/models/flow_module.py
"""

import contextlib
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import Tensor

from astrafold.config.hydra_schema import HydraConfig
from astrafold.data import so3_utils
from astrafold.data.contacts import random_mask_tokens
from astrafold.data.interpolant import Interpolant
from astrafold.metrics import kabsch_rmsd
from astrafold.models.flow_model import FlowModel

# The model predicts the first 9 torsion angles of each nucleotide.
NUM_TORSIONS = 9


class FlowModule(LightningModule):
    def __init__(self, cfg: HydraConfig):
        super().__init__()
        self._exp_cfg = cfg.experiment
        self._interpolant_cfg = cfg.interpolant

        self.model = FlowModel(cfg.model)  # predicts the clean structure x1
        self.interpolant = Interpolant(cfg.interpolant)

        self.save_hyperparameters()

    def corrupt_batch(
        self, batch: dict[str, Tensor], t: Tensor | None = None
    ) -> dict[str, Tensor]:
        """
        Builds the model input for a clean batch: noises the frames to flow time `t`
        (sampled uniformly if None) and masks a random subset of the contact tokens.

        On top of the interpolant's output, the returned batch holds:
            contact_tokens   : the partially masked tokens (model input)
            contact_tokens_1 : the clean tokens (prediction target)
            token_mask       : True where a token was masked
        """
        self.interpolant.set_device(batch["res_mask"].device)
        noisy_batch = self.interpolant.corrupt_batch(batch, t=t)

        tokens = noisy_batch["contact_tokens"]
        masked_tokens, token_mask = random_mask_tokens(
            tokens,
            noisy_batch["res_mask"],
            mask_probability=self._exp_cfg.training.token_mask_probability,
        )
        noisy_batch["contact_tokens_1"] = tokens
        noisy_batch["contact_tokens"] = masked_tokens
        noisy_batch["token_mask"] = token_mask
        return noisy_batch

    def _self_condition(self, noisy_batch: dict[str, Tensor]) -> None:
        """Adds the model's own x1 prediction to the batch as an extra input."""
        with torch.no_grad():
            model_sc = self.model(noisy_batch)
        noisy_batch["trans_sc"] = model_sc["pred_trans"]

    def model_step(
        self, noisy_batch: dict[str, Tensor]
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """
        Runs the model on a batch from `corrupt_batch` and computes the losses between
        the predicted and the ground truth structure.

        Returns:
            The losses (per example, except for the batch-level `token_loss` and
            `token_accuracy`) and the model output.
        """
        training_cfg = self._exp_cfg.training
        loss_mask = noisy_batch["res_mask"]
        if training_cfg.min_plddt_mask is not None:
            plddt_mask = noisy_batch["res_plddt"] > training_cfg.min_plddt_mask
            loss_mask *= plddt_mask
        num_batch, num_res = loss_mask.shape
        num_frames = torch.sum(loss_mask, dim=-1)

        # Ground truth
        gt_trans_1 = noisy_batch["trans_1"]
        gt_rotmats_1 = noisy_batch["rotmats_1"]
        gt_torsions_1 = noisy_batch["torsion_angles_sin_cos"][:, :, :NUM_TORSIONS, :]
        gt_tokens = noisy_batch["contact_tokens_1"]
        token_mask = noisy_batch["token_mask"]
        rotmats_t = noisy_batch["rotmats_t"]
        gt_rot_vf = so3_utils.calc_rot_vf(rotmats_t, gt_rotmats_1.type(torch.float32))

        # The vector field losses are scaled by 1 / (1 - t), clipped close to t = 1.
        t = noisy_batch["t"]
        norm_scale = 1 - torch.clamp(t[..., None], max=training_cfg.t_normalize_clip)

        # Predictions
        model_output: dict[str, Tensor] = self.model(noisy_batch)
        pred_trans_1 = model_output["pred_trans"]
        pred_rotmats_1 = model_output["pred_rotmats"]
        pred_torsions_1 = model_output["pred_torsions"].reshape(
            num_batch, num_res, NUM_TORSIONS, 2
        )
        token_logits = model_output["token_logits"]
        pred_rots_vf = so3_utils.calc_rot_vf(rotmats_t, pred_rotmats_1)

        # Translation vector field loss
        trans_error = (
            (gt_trans_1 - pred_trans_1) / norm_scale * training_cfg.trans_scale
        )
        trans_loss = (
            training_cfg.translation_loss_weight
            * torch.sum(trans_error**2 * loss_mask[..., None], dim=(-1, -2))
            / (num_frames * 3)
        )

        # Rotation vector field loss
        rots_vf_error = (gt_rot_vf - pred_rots_vf) / norm_scale
        rots_vf_loss = (
            training_cfg.rotation_loss_weights
            * torch.sum(rots_vf_error**2 * loss_mask[..., None], dim=(-1, -2))
            / (num_frames * 3)
        )

        # Contact token loss and accuracy, on the masked tokens only
        token_loss = F.cross_entropy(token_logits[token_mask], gt_tokens[token_mask])
        with torch.no_grad():
            pred_tokens = token_logits.argmax(dim=-1)
            token_accuracy = (
                (pred_tokens[token_mask] == gt_tokens[token_mask]).float().mean()
            )

        # Torsion angle loss
        tors_loss = (
            training_cfg.tors_loss_scale
            * torch.sum(
                torch.linalg.norm(pred_torsions_1 - gt_torsions_1, dim=-1) ** 2
                * loss_mask[..., None],
                dim=(-1, -2),
            )
            / (num_frames * NUM_TORSIONS)
        )

        se3_vf_loss = trans_loss + rots_vf_loss
        # The torsion loss only counts once the structure is mostly denoised.
        auxiliary_loss = tors_loss * (t[:, 0] > training_cfg.aux_loss_t_pass)
        auxiliary_loss *= training_cfg.aux_loss_weight

        # Zero NaN losses with torch.where rather than zeros_like: the result keeps its
        # autograd graph, so every parameter still gets a gradient and DDP does not fail
        # on "unused parameters". Any NaN gradients are then caught by
        # on_before_optimizer_step.
        if torch.isnan(auxiliary_loss).any():
            print("NaN loss in aux_loss")
            auxiliary_loss = torch.where(
                torch.isnan(auxiliary_loss), 0.0, auxiliary_loss
            )

        if torch.isnan(se3_vf_loss).any():
            print("NaN loss in se3_vf_loss")
            se3_vf_loss = torch.where(torch.isnan(se3_vf_loss), 0.0, se3_vf_loss)

        losses = {
            "trans_loss": trans_loss,
            "auxiliary_loss": auxiliary_loss,
            "rots_vf_loss": rots_vf_loss,
            "se3_vf_loss": se3_vf_loss,
            "torsion_loss": tors_loss,
            "token_loss": token_loss,
            "token_accuracy": token_accuracy,
        }
        return losses, model_output

    def _total_loss(self, mean_losses: dict[str, Tensor]) -> Tensor:
        """Combines the batch-mean losses into the optimized (and monitored) loss."""
        training_cfg = self._exp_cfg.training
        return (
            mean_losses[training_cfg.loss]
            + mean_losses["auxiliary_loss"]
            + training_cfg.token_loss_weight * mean_losses["token_loss"]
        )

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        """
        Performs one iteration of SE(3) flow matching and returns the total training
        loss, combined from the losses computed in `model_step`.
        """
        step_start_time = time.time()
        smoke_test_seed = self._exp_cfg.training.smoke_test_seed
        if smoke_test_seed is not None:
            random.seed(smoke_test_seed)
            np.random.seed(smoke_test_seed)
            torch.manual_seed(smoke_test_seed)

        noisy_batch = self.corrupt_batch(batch)
        if self._interpolant_cfg.self_condition and random.random() > 0.5:
            self._self_condition(noisy_batch)

        batch_losses, _ = self.model_step(noisy_batch)
        num_batch = batch_losses["torsion_loss"].shape[0]
        mean_losses = {k: torch.mean(v) for k, v in batch_losses.items()}

        for name, value in mean_losses.items():
            self._log_scalar(
                f"train/{name}",
                value,
                on_step=True,
                prog_bar=name == "token_accuracy",
                batch_size=num_batch,
            )
        self._log_scalar(
            "train/t",
            torch.mean(noisy_batch["t"]),
            on_step=True,
            prog_bar=False,
            batch_size=num_batch,
        )

        step_time = time.time() - step_start_time
        self._log_scalar("train/eps", num_batch / step_time)

        train_loss = self._total_loss(mean_losses)
        self._log_scalar("loss", train_loss, on_step=True, batch_size=num_batch)

        return train_loss

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """
        Skips the update when any gradient is non-finite, so one bad step cannot turn
        all weights into NaN. Runs after the DDP all-reduce, so every rank sees the
        same gradients and makes the same decision. The gradients are zeroed rather
        than the step dropped, so AdamW still applies its decay and momentum.
        """
        grads = [p.grad for p in self.parameters() if p.grad is not None]
        finite = torch.stack([torch.isfinite(g).all() for g in grads]).all()
        if not finite:
            print(f"Non-finite gradients at step {self.global_step}: update skipped")
            for g in grads:
                g.zero_()
        self.log(
            "train/skipped_steps",
            (~finite).float(),
            on_step=False,
            on_epoch=True,
            reduce_fx="sum",
            sync_dist=False,
        )

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        """
        Computes the training losses on a held-out batch at each flow time in
        `validation.t_values`, plus the RMSD between the model's one-step prediction
        of x1 and the ground truth (frame origins, after Kabsch alignment).

        Noise and token masks are fixed per batch by `validation.seed` (and dropout
        is off in eval mode), so the metrics only change when the model does. Logged
        per epoch as `valid/<metric>` (mean over the t grid) and
        `valid/<metric>_t<t>` for the loss and the RMSD.
        """
        val_cfg = self._exp_cfg.validation
        res_mask = batch["res_mask"]
        device = res_mask.device
        num_batch = res_mask.shape[0]

        per_t_metrics: list[dict[str, Tensor]] = []
        with _seeded_rng(val_cfg.seed + batch_idx, device):
            for t_value in val_cfg.t_values:
                t = torch.full((num_batch, 1), float(t_value), device=device)
                noisy_batch = self.corrupt_batch(batch, t=t)

                # Training self-conditions half of its steps; validation always does.
                if self._interpolant_cfg.self_condition:
                    self._self_condition(noisy_batch)

                batch_losses, model_output = self.model_step(noisy_batch)
                metrics = {k: torch.mean(v) for k, v in batch_losses.items()}
                metrics["loss"] = self._total_loss(metrics)
                metrics["x1_rmsd"] = torch.mean(
                    kabsch_rmsd(
                        model_output["pred_trans"], noisy_batch["trans_1"], res_mask
                    )
                )
                per_t_metrics.append(metrics)

                for name in ("loss", "x1_rmsd"):
                    self._log_scalar(
                        f"valid/{name}_t{t_value:g}",
                        metrics[name],
                        prog_bar=False,
                        batch_size=num_batch,
                    )

        for name in per_t_metrics[0]:
            self._log_scalar(
                f"valid/{name}",
                torch.stack([m[name] for m in per_t_metrics]).mean(),
                prog_bar=name == "loss",
                batch_size=num_batch,
            )

    def _log_scalar(
        self,
        key,
        value,
        on_step=False,
        on_epoch=True,
        prog_bar=True,
        batch_size=None,
        sync_dist=True,
        rank_zero_only=False,
    ):
        if sync_dist and rank_zero_only:
            raise ValueError("Unable to sync dist when rank_zero_only=True")
        if sync_dist:
            # Lightning stores the accumulated batch size as an int and averages it
            # across ranks, which truncates when ranks have different batch sizes
            # (e.g. epoch accuracy > 1). Equal weights per step avoid this.
            batch_size = 1
        self.log(
            key,
            value,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=prog_bar,
            batch_size=batch_size,
            sync_dist=sync_dist,
            rank_zero_only=rank_zero_only,
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(
            params=self.model.parameters(), **self._exp_cfg.optimizer
        )


@contextlib.contextmanager
def _seeded_rng(seed: int, device: torch.device):
    """
    Seeds torch, numpy and python RNGs for the duration of the block and restores
    their previous states afterwards, so validation doesn't shift the training noise.
    """
    np_state = np.random.get_state()
    py_state = random.getstate()
    cuda_devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        try:
            yield
        finally:
            np.random.set_state(np_state)
            random.setstate(py_state)

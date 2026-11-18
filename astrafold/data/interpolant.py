"""
Noising process of the flow: interpolates frames between noise (t = 0) and data (t = 1),
and the sampler that integrates the learned flow back from noise to data.

Code adapted from
https://github.com/microsoft/protein-frame-flow/blob/main/data/interpolant.py
"""

import copy

import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation
from torch import Tensor

from astrafold.config.hydra_schema import InterpolantConfig
from astrafold.data import so3_utils
from astrafold.data import utils as du
from astrafold.data.contacts import MASK, unmask_tokens


def _centered_gaussian(num_batch, num_res, device):
    # returns randomly-sampled zero-centered Gaussian noise for translations
    noise = torch.randn(num_batch, num_res, 3, device=device)
    return noise - torch.mean(noise, dim=-2, keepdims=True)


def _uniform_so3(num_batch, num_res, device):
    # returns uniformly-sampled rotations (from U(SO(3)))
    return torch.tensor(
        Rotation.random(num_batch * num_res).as_matrix(),  # random rotation
        device=device,
        dtype=torch.float32,
    ).reshape(num_batch, num_res, 3, 3)


def seeded_generators(seeds: list[int]) -> list[torch.Generator]:
    """One CPU generator per sample, so a sample's noise depends only on its seed."""
    return [torch.Generator().manual_seed(int(seed)) for seed in seeds]


def draw_per_sample(generators, sample_fn, device) -> Tensor:
    """Stacks `sample_fn(generator)` over the generators: [B, ...] on `device`."""
    return torch.stack([sample_fn(g) for g in generators]).to(device)


def _trans_diffuse_mask(trans_t, trans_1, diffuse_mask):
    return trans_t * diffuse_mask[..., None] + trans_1 * (1 - diffuse_mask[..., None])


def _rots_diffuse_mask(rotmats_t, rotmats_1, diffuse_mask):
    return rotmats_t * diffuse_mask[..., None, None] + rotmats_1 * (
        1 - diffuse_mask[..., None, None]
    )


class Interpolant:
    def __init__(self, cfg: InterpolantConfig):
        self._cfg = cfg
        self._rots_cfg = cfg.rots
        self._trans_cfg = cfg.trans
        self._sample_cfg = cfg.sampling
        self._igso3 = None

    @property
    def igso3(self):
        if self._igso3 is None:
            sigma_grid = torch.linspace(0.1, 1.5, 1000)
            self._igso3 = so3_utils.SampleIGSO3(1000, sigma_grid, cache_dir=".cache")
        # Keep the lookup tables on the training device so sampling runs there.
        return self._igso3.to(self._device)

    def set_device(self, device):
        self._device = device

    def sample_t(self, num_batch):
        """
        Sample random timestep for flow matching
        """
        t = torch.rand(num_batch, device=self._device)
        return t * (1 - 2 * self._cfg.min_t) + self._cfg.min_t

    def _corrupt_trans(self, trans_1, t, res_mask):
        """
        Returns optimal transport-based interpolated "flow" for translations
        """
        trans_nm_0 = _centered_gaussian(*res_mask.shape, self._device)

        trans_0 = trans_nm_0 * du.NM_TO_ANG_SCALE
        trans_0 = self._batch_ot(trans_0, trans_1, res_mask)
        trans_t = (1 - t[..., None]) * trans_0 + t[
            ..., None
        ] * trans_1  # flow: t*x1 + (1-t)*x0
        trans_t = _trans_diffuse_mask(trans_t, trans_1, res_mask)
        return trans_t * res_mask[..., None]

    def _batch_ot(self, trans_0, trans_1, res_mask):
        """
        Use optimal transport to create the unified "flow" for
        translations by computing a cost matrix
        """
        num_batch, num_res = trans_0.shape[:2]
        noise_idx, gt_idx = torch.where(torch.ones(num_batch, num_batch))
        batch_nm_0 = trans_0[noise_idx]
        batch_nm_1 = trans_1[gt_idx]
        batch_mask = res_mask[gt_idx]
        aligned_nm_0, aligned_nm_1, _ = du.batch_align_structures(
            batch_nm_0, batch_nm_1, mask=batch_mask
        )
        aligned_nm_0 = aligned_nm_0.reshape(num_batch, num_batch, num_res, 3)
        aligned_nm_1 = aligned_nm_1.reshape(num_batch, num_batch, num_res, 3)

        # Compute cost matrix of aligned noise to ground truth
        batch_mask = batch_mask.reshape(num_batch, num_batch, num_res)
        cost_matrix = torch.sum(
            torch.linalg.norm(aligned_nm_0 - aligned_nm_1, dim=-1), dim=-1
        ) / torch.sum(batch_mask, dim=-1)
        noise_perm, gt_perm = linear_sum_assignment(du.to_numpy(cost_matrix))
        return aligned_nm_0[(tuple(gt_perm), tuple(noise_perm))]

    def _corrupt_rotmats(self, rotmats_1, t, res_mask):
        """
        Returns interpolated "flow" for rotations without optimal transport
        """
        num_batch, num_res = res_mask.shape
        noisy_rotmats = self.igso3.sample(  # sample from IGSO(3)
            torch.tensor([1.5], device=self._device), num_batch * num_res
        )
        noisy_rotmats = noisy_rotmats.reshape(num_batch, num_res, 3, 3)

        rotmats_0 = torch.einsum("...ij,...jk->...ik", rotmats_1, noisy_rotmats)

        # compute "flow" for rotations using log/exp maps
        rotmats_t = so3_utils.geodesic_t(t[..., None], rotmats_1, rotmats_0)
        identity = torch.eye(3, device=self._device)
        rotmats_t = rotmats_t * res_mask[..., None, None] + identity[None, None] * (
            1 - res_mask[..., None, None]
        )
        return _rots_diffuse_mask(rotmats_t, rotmats_1, res_mask)

    def corrupt_batch(
        self, batch: dict[str, Tensor], t: Tensor | None = None
    ) -> dict[str, Tensor]:
        """
        Params:
            batch (dict) : a dictionary where each key is a tensor name and each value is the corresponding tensor of Rigid objects (representing the RNA frames).
            t (Tensor, optional) : [B, 1] flow times to corrupt to; sampled uniformly if None.

        Remarks:
            This method takes in a clean ground truth batch (x_1) and returns the interpolated (noised) batch (x_t).

        Returns:
            batch of corrupted/noisy/interpolated tensors
        """

        noisy_batch = copy.deepcopy(batch)

        trans_1 = batch["trans_1"]  # # [B, N, 3] (in Angstrom)
        rotmats_1 = batch["rotmats_1"]  # [B, N, 3, 3]
        res_mask = batch["res_mask"]  # [B, N]
        num_batch, _ = res_mask.shape

        # get random timestep in [0, 1] for flow matching
        if t is None:
            t = self.sample_t(num_batch)[:, None]  # [B, 1]
        noisy_batch["t"] = t

        # apply corruptions to translations
        trans_t = self._corrupt_trans(trans_1, t, res_mask)
        noisy_batch["trans_t"] = trans_t

        # apply corruptions to rotations
        rotmats_t = self._corrupt_rotmats(rotmats_1, t, res_mask)
        noisy_batch["rotmats_t"] = rotmats_t

        return noisy_batch

    def rot_sample_kappa(self, t):
        if self._rots_cfg.sample_schedule == "exp":
            return 1 - torch.exp(-t * self._rots_cfg.exp_rate)
        elif self._rots_cfg.sample_schedule == "linear":
            return t
        else:
            raise ValueError(f"Invalid schedule: {self._rots_cfg.sample_schedule}")

    def _trans_euler_step(self, d_t, t, trans_1, trans_t):
        """
        Perform Euler ODESolve step for translations
        """
        trans_vf = (trans_1 - trans_t) / (1 - t)
        return trans_t + trans_vf * d_t

    def _rots_euler_step(self, d_t, t, rotmats_1, rotmats_t):
        """
        Perform Euler ODESolve step for rotations based on prescribed sampling schedule
        """
        if self._rots_cfg.sample_schedule == "linear":
            scaling = 1 / (1 - t)
        elif self._rots_cfg.sample_schedule == "exp":
            scaling = self._rots_cfg.exp_rate
        else:
            raise ValueError(
                f"Unknown sample schedule {self._rots_cfg.sample_schedule}"
            )

        return so3_utils.geodesic_t(scaling * d_t, rotmats_1, rotmats_t)

    def euler_step(
        self,
        d_t: float | Tensor,
        t: float | Tensor,
        model_out: dict[str, Tensor],
        trans_t: Tensor,
        rotmats_t: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        One Euler step of the flow from t to t + d_t towards the model's x1
        prediction. `t` and `d_t` are scalars or [B, 1, 1] tensors (one flow time
        per sample); t must be < 1.
        """
        trans = self._trans_euler_step(d_t, t, model_out["pred_trans"], trans_t)
        rotmats = self._rots_euler_step(d_t, t, model_out["pred_rotmats"], rotmats_t)
        return trans, rotmats

    def sample_prior(
        self, generators: list[torch.Generator], num_res: int, device: torch.device
    ) -> tuple[Tensor, Tensor]:
        """
        Draws the frames at t = min_t, one sample per generator: centered Gaussian
        translations [B, N, 3] (in Angstrom) and uniformly random rotations
        [B, N, 3, 3].
        """
        trans = draw_per_sample(
            generators, lambda g: torch.randn(num_res, 3, generator=g), device
        )
        trans = trans - trans.mean(dim=-2, keepdim=True)
        trans = trans * du.NM_TO_ANG_SCALE
        # Normalized Gaussian quaternions are uniform on SO(3).
        quats = draw_per_sample(
            generators, lambda g: torch.randn(num_res, 4, generator=g), device
        )
        rotmats = torch.tensor(
            Rotation.from_quat(quats.cpu().numpy().reshape(-1, 4)).as_matrix(),
            dtype=torch.float32,
            device=device,
        ).reshape(len(generators), num_res, 3, 3)
        return trans, rotmats

    @torch.no_grad()
    def sample(
        self,
        model: torch.nn.Module,
        onehot: Tensor,
        seeds: list[int],
        num_timesteps: int | None = None,
    ) -> dict[str, Tensor]:
        """
        Generates one structure per seed for a single sequence: integrates the frames
        from noise (t = min_t) to data (t = 1) with Euler steps, while the contact
        tokens go from fully masked to fully revealed. In every step the model
        predicts the clean structure and the tokens; a masked token is then revealed
        with probability d_t / (1 - t), so that on average a fraction 1 - t of the
        tokens is masked at flow time t.

        Each sample draws all its randomness from its own seed, so a sample does not
        depend on which other samples share its batch.

        Params:
            model         : predicts x1 and the token logits (in eval mode)
            onehot        : [N, 4] one-hot sequence, on the device to sample on
            seeds         : one seed per sample (B = len(seeds))
            num_timesteps : number of flow times; defaults to `sampling.num_timesteps`

        Returns:
            trans          : [B, N, 3] frame origins (in Angstrom)
            rotmats        : [B, N, 3, 3] frame orientations
            torsions       : [B, N, 9, 2] torsion angles as (sin, cos)
            contact_tokens : [B, N] the generated contact tokens
        """
        if num_timesteps is None:
            num_timesteps = self._sample_cfg.num_timesteps
        if num_timesteps < 2:
            raise ValueError("Sampling needs at least 2 timesteps")

        device = onehot.device
        self.set_device(device)
        num_batch, num_res = len(seeds), onehot.shape[0]
        generators = seeded_generators(seeds)

        def draw(sample_fn) -> Tensor:
            return draw_per_sample(generators, sample_fn, device)

        trans_t, rotmats_t = self.sample_prior(generators, num_res, device)
        tokens = torch.full((num_batch, num_res), MASK, device=device)

        batch = {
            "res_mask": torch.ones(num_batch, num_res, device=device),
            "onehot": onehot[None].expand(num_batch, -1, -1),
        }

        def predict(t: float) -> dict[str, Tensor]:
            batch["trans_t"] = trans_t
            batch["rotmats_t"] = rotmats_t
            batch["contact_tokens"] = tokens
            batch["t"] = torch.full((num_batch, 1), t, device=device)
            return model(batch)

        ts = torch.linspace(self._cfg.min_t, 1.0, num_timesteps).tolist()
        t_1 = ts[0]
        for t_2 in ts[1:]:
            model_out = predict(t_1)
            if self._cfg.self_condition:
                batch["trans_sc"] = model_out["pred_trans"]

            d_t = t_2 - t_1
            trans_t, rotmats_t = self.euler_step(
                d_t, t_1, model_out, trans_t, rotmats_t
            )
            tokens = unmask_tokens(
                tokens,
                model_out["token_logits"],
                # Reveals all remaining tokens in the last step (t_2 = 1).
                reveal_probability=d_t / (1 - t_1),
                reveal_noise=draw(lambda g: torch.rand(num_res, generator=g)),
                value_noise=draw(lambda g: torch.rand(num_res, generator=g)),
            )
            t_1 = t_2

        # The integration stops at t = 1; the final structure is the model's
        # prediction there, given the fully revealed tokens.
        model_out = predict(t_1)
        return {
            "trans": model_out["pred_trans"],
            "rotmats": model_out["pred_rotmats"],
            "torsions": model_out["pred_torsions"].reshape(num_batch, num_res, -1, 2),
            "contact_tokens": tokens,
        }

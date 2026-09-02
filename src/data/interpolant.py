import torch, copy
from src.data import so3_utils
from src.data import utils as du

from scipy.spatial.transform import Rotation
from scipy.optimize import linear_sum_assignment

def _centered_gaussian(num_batch, num_res, device):
    # returns randomly-sampled zero-centered Gaussian noise for translations
    noise = torch.randn(num_batch, num_res, 3, device=device)
    return noise - torch.mean(noise, dim=-2, keepdims=True)

def _uniform_so3(num_batch, num_res, device):
    # returns uniformly-sampled rotations (from U(SO(3)))
    return torch.tensor(
        Rotation.random(num_batch*num_res).as_matrix(), # random rotation
        device=device,
        dtype=torch.float32,
    ).reshape(num_batch, num_res, 3, 3)

def _trans_diffuse_mask(trans_t, trans_1, diffuse_mask):
    return trans_t * diffuse_mask[..., None] + trans_1 * (1 - diffuse_mask[..., None])

def _rots_diffuse_mask(rotmats_t, rotmats_1, diffuse_mask):
    return (
        rotmats_t * diffuse_mask[..., None, None]
        + rotmats_1 * (1 - diffuse_mask[..., None, None])
    )

class Interpolant:
    def __init__(self, cfg):
        self._cfg = cfg
        self._rots_cfg = cfg.rots
        self._trans_cfg = cfg.trans
        self._sample_cfg = cfg.sampling
        self._igso3 = None

    @property
    def igso3(self):
        if self._igso3 is None:
            sigma_grid = torch.linspace(0.1, 1.5, 1000)
            self._igso3 = so3_utils.SampleIGSO3(
                1000, sigma_grid, cache_dir='.cache')
        return self._igso3

    def set_device(self, device):
        self._device = device

    def sample_t(self, num_batch):
       """
       Sample random timestep for flow matching
       """
       t = torch.rand(num_batch, device=self._device)
       return t * (1 - 2*self._cfg.min_t) + self._cfg.min_t

    def _corrupt_trans(self, trans_1, t, res_mask):
        """
        Returns optimal transport-based interpolated "flow" for translations
        """
        trans_nm_0 = _centered_gaussian(*res_mask.shape, self._device)

        trans_0 = trans_nm_0 * du.NM_TO_ANG_SCALE
        trans_0 = self._batch_ot(trans_0, trans_1, res_mask)
        trans_t = (1 - t[..., None]) * trans_0 + t[..., None] * trans_1 # flow: t*x1 + (1-t)*x0
        trans_t = _trans_diffuse_mask(trans_t, trans_1, res_mask)
        return trans_t * res_mask[..., None]
    
    def _batch_ot(self, trans_0, trans_1, res_mask):
        """
        Use optimal transport to create the unified "flow" for 
        translations by computing a cost matrix
        """
        num_batch, num_res = trans_0.shape[:2]
        noise_idx, gt_idx = torch.where(
            torch.ones(num_batch, num_batch))
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
        noisy_rotmats = self.igso3.sample( # sample from IGSO(3)
            torch.tensor([1.5]),
            num_batch*num_res
        ).to(self._device)
        noisy_rotmats = noisy_rotmats.reshape(num_batch, num_res, 3, 3)

        rotmats_0 = torch.einsum("...ij,...jk->...ik", rotmats_1, noisy_rotmats)

        # compute "flow" for rotations using log/exp maps
        rotmats_t = so3_utils.geodesic_t(t[..., None], rotmats_1, rotmats_0)
        identity = torch.eye(3, device=self._device)
        rotmats_t = (
            rotmats_t * res_mask[..., None, None]
            + identity[None, None] * (1 - res_mask[..., None, None])
        )
        return _rots_diffuse_mask(rotmats_t, rotmats_1, res_mask)

    def corrupt_batch(self, batch):
        """
        Params:
            batch (dict) : a dictionary where each key is a tensor name and each value is the corresponding tensor of Rigid objects (representing the RNA frames).

        Remarks:
            This method takes in a clean ground truth batch (x_1) and adds returns the interpolated (noised) batch (x_t).

        Returns:
            batch of corrupted/noisy/interpolated tensors
        """

        noisy_batch = copy.deepcopy(batch)

        trans_1 = batch['trans_1'] # # [B, N, 3] (in Angstrom)
        rotmats_1 = batch['rotmats_1'] # [B, N, 3, 3]
        res_mask = batch['res_mask'] # [B, N]
        num_batch, _ = res_mask.shape

        # get random timestep in [0, 1] for flow matching
        t = self.sample_t(num_batch)[:, None] # [B, 1]
        noisy_batch['t'] = t

        # apply corruptions to translations
        trans_t = self._corrupt_trans(trans_1, t, res_mask)
        noisy_batch['trans_t'] = trans_t

        # apply corruptions to rotations
        rotmats_t = self._corrupt_rotmats(rotmats_1, t, res_mask)
        noisy_batch['rotmats_t'] = rotmats_t
        
        return noisy_batch
    
    def rot_sample_kappa(self, t):
        if self._rots_cfg.sample_schedule == 'exp':
            return 1 - torch.exp(-t*self._rots_cfg.exp_rate)
        elif self._rots_cfg.sample_schedule == 'linear':
            return t
        else:
            raise ValueError(
                f'Invalid schedule: {self._rots_cfg.sample_schedule}')

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
        if self._rots_cfg.sample_schedule == 'linear':
            scaling = 1 / (1 - t)
        elif self._rots_cfg.sample_schedule == 'exp':
            scaling = self._rots_cfg.exp_rate
        else:
            raise ValueError(f'Unknown sample schedule {self._rots_cfg.sample_schedule}')
        
        return so3_utils.geodesic_t(scaling * d_t, rotmats_1, rotmats_t)
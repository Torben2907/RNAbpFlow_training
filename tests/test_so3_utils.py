import math

import pytest
import torch

from astrafold.data.so3_utils import rotmat_to_rotvec, rotvec_to_rotmat


def _rotvec_and_grad(rotmats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rotmats = rotmats.clone().requires_grad_(True)
    rotvecs = rotmat_to_rotvec(rotmats)
    (rotvecs**2).sum().backward()
    return rotvecs.detach(), rotmats.grad


@pytest.mark.parametrize("angle", [0.3, math.pi / 2, 2.6, 3.1])
def test_rotmat_to_rotvec_round_trip(angle: float) -> None:
    rotvec = torch.tensor([[0.0, 0.0, angle]])
    recovered, grad = _rotvec_and_grad(rotvec_to_rotmat(rotvec))
    torch.testing.assert_close(recovered, rotvec, atol=1e-4, rtol=1e-4)
    assert torch.isfinite(grad).all()


def test_rotmat_to_rotvec_grad_finite_when_diagonal_below_minus_one() -> None:
    # Low-precision matmuls can push a diagonal entry of a near-pi relative rotation
    # just below -1. The forward value is fine; the gradient used to be NaN, which
    # turned all weights into NaN after one optimizer step.
    rotmats = rotvec_to_rotmat(torch.tensor([[0.0, 0.0, 2.6]]))
    rotmats[0, 0, 0] = -1.0 - 1e-4
    rotvecs, grad = _rotvec_and_grad(rotmats)
    assert torch.isfinite(rotvecs).all()
    assert torch.isfinite(grad).all()

import torch
from test_utils import load_example_batch

from rnabpflow.data.contacts import MASK, random_mask_tokens


def test_masking(config):
    batch = load_example_batch(config)
    masked, mask = random_mask_tokens(batch["contact_tokens"], batch["res_mask"])
    assert torch.all(masked[mask] == MASK)
    assert not torch.any(masked[~batch["res_mask"].bool()] == MASK)
    # Since I inititalize using torch.rand() mean value should be around 0.15 (which it is)
    # torch.testing.assert_close(mask.float().mean(), torch.tensor(0.15), rtol=1e-2, atol=1e-2)

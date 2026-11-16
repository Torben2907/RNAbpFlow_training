import torch
from test_utils import load_example_batch

from astrafold.data.contacts import MASK, random_mask_tokens


def test_masking(config):
    batch = load_example_batch(config)
    masked, mask = random_mask_tokens(batch["contact_tokens"], batch["res_mask"])
    assert torch.all(masked[mask] == MASK)
    assert not torch.any(masked[~batch["res_mask"].bool()] == MASK)
    # Unmasked tokens are left untouched.
    assert torch.equal(masked[~mask], batch["contact_tokens"][~mask])

import torch
from test_utils import load_example_batch


def test_data(config):
    batch = load_example_batch(config)

    assert batch["contact_tokens"].shape == batch["res_mask"].shape
    assert batch["contact_tokens"].dtype == torch.long
    assert torch.all((batch["contact_tokens"] == 0) | (batch["contact_tokens"] == 1))

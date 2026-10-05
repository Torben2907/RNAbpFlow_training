import torch
from test_utils import load_example_batch


def test_data():
    batch = load_example_batch()

    assert batch["contact_tokens"].shape == batch["res_mask"].shape
    assert batch["contact_tokens"].dtype == torch.long
    assert torch.all((batch["contact_tokens"] == 0) | (batch["contact_tokens"] == 1))


if __name__ == "__main__":
    example_batch = load_example_batch()
    print(
        "Loaded batch:",
        {
            key: tuple(value.shape)
            for key, value in example_batch.items()
            if torch.is_tensor(value)
        },
    )

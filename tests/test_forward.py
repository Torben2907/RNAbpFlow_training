import torch
from test_utils import load_example_batch
from torch import Tensor

from rnabpflow.config.hydra_schema import HydraConfig
from rnabpflow.data.contacts import random_mask_tokens
from rnabpflow.data.interpolant import Interpolant
from rnabpflow.models.flow_model import FlowModel
from rnabpflow.models.flow_module import FlowModule


def test_shapes(config: HydraConfig):
    batch = load_example_batch(config)
    B, N = batch["contact_tokens"].size()

    interpolant = Interpolant(config.interpolant)
    interpolant.set_device(batch["contact_tokens"].device)
    model = FlowModel(config.model)
    noisy_batch = interpolant.corrupt_batch(batch)

    masked_tokens, token_mask = random_mask_tokens(
        noisy_batch["contact_tokens"], noisy_batch["res_mask"]
    )

    noisy_batch["contact_tokens"] = masked_tokens
    noisy_batch["token_mask"] = token_mask

    output: dict[str, Tensor] = model(noisy_batch)

    assert output["pred_trans"].shape == (B, N, 3)
    assert output["pred_rotmats"].shape == (B, N, 3, 3)
    assert output["pred_torsions"].shape == (B, N, 9, 2)
    assert output["pair_feat"].shape == (B, N, N, 3)
    assert output["token_logits"].shape == (B, N, 2)


def test_losses_are_not_NaN(config: HydraConfig):
    batch = load_example_batch(config)
    interpolant = Interpolant(config.interpolant)
    interpolant.set_device(batch["contact_tokens"].device)
    noisy_batch = interpolant.corrupt_batch(batch)
    flow_module = FlowModule(config)

    losses = flow_module.model_step(noisy_batch)

    for k, v in losses.items(): 
        assert not torch.isnan(v).any(), f"{k} contains NaN values"
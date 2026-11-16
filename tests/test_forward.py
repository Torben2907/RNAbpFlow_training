import copy
import random

import torch
from test_utils import load_example_batch
from torch import Tensor

from astrafold.config.hydra_schema import HydraConfig
from astrafold.data.contacts import MASK, random_mask_tokens
from astrafold.data.interpolant import Interpolant
from astrafold.models.flow_model import FlowModel
from astrafold.models.flow_module import FlowModule


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
    assert output["token_logits"].shape == (B, N, 2)


def test_losses_are_not_NaN(config: HydraConfig):
    batch = load_example_batch(config)
    flow_module = FlowModule(config)
    noisy_batch = flow_module.corrupt_batch(batch)

    losses, _ = flow_module.model_step(noisy_batch)

    for k, v in losses.items():
        assert not torch.isnan(v).any(), f"{k} contains NaN values"


def test_self_conditioning_pass_sees_masked_tokens(config: HydraConfig, monkeypatch):
    cfg = copy.deepcopy(config)
    cfg.interpolant.self_condition = True
    batch = load_example_batch(cfg)
    flow_module = FlowModule(cfg)
    flow_module.log = lambda *args, **kwargs: None
    monkeypatch.setattr(random, "random", lambda: 1.0)  # always self-condition

    seen_tokens: list[Tensor] = []
    flow_module.model.register_forward_pre_hook(
        lambda module, args: seen_tokens.append(args[0]["contact_tokens"].clone())
    )
    flow_module.training_step(batch, 0)

    # Both the self-conditioning pass and the main pass get the same masked tokens,
    # never the ground truth.
    sc_tokens, main_tokens = seen_tokens
    assert torch.equal(sc_tokens, main_tokens)
    assert (sc_tokens == MASK).any()
    assert not torch.equal(sc_tokens, batch["contact_tokens"])

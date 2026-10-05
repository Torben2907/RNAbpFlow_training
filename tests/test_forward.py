from test_utils import load_example_batch
from torch import Tensor

from rnabpflow.config.hydra_schema import HydraConfig
from rnabpflow.data.contacts import random_mask_tokens
from rnabpflow.data.interpolant import Interpolant
from rnabpflow.models.flow_model import FlowModel


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

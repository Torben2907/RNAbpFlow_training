import torch
from torch import Tensor

CONTACT = 1
NO_CONTACT = 0
MASK = 2

NUM_TOKEN_STATES = 3


def make_contact_tokens(
    c1_positions: Tensor,
    residue_mask: Tensor,
    distance_threshold: float = 8.0,
    min_sequence_separation: int = 3,
) -> Tensor:

    d = torch.cdist(c1_positions, c1_positions)
    n = d.shape[-1]

    idx = torch.arange(n, device=d.device)
    seq_sep = (idx[:, None] - idx[None, :]).abs()

    contact = (
        (d < distance_threshold)
        & (seq_sep > min_sequence_separation)
        & residue_mask[:, None].bool()
        & residue_mask[None, :].bool()
    )

    has_contact = contact.any(dim=-1)

    tokens = torch.where(
        has_contact,
        torch.full_like(has_contact, fill_value=CONTACT, dtype=torch.long),
        torch.full_like(has_contact, fill_value=NO_CONTACT, dtype=torch.long),
    )

    return tokens


def random_mask_tokens(
    tokens: Tensor, res_mask: Tensor, mask_probability: float = 0.15
) -> tuple[Tensor, Tensor]:
    random_mask = (torch.rand_like(tokens.float()) < mask_probability) & res_mask.bool()

    masked_tokens = tokens.clone()
    masked_tokens[random_mask] = MASK

    return masked_tokens, random_mask

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
    N = d.shape[-1]

    idx = torch.arange(N, device=d.device)
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
    tokens: Tensor, res_mask: Tensor, mask_probability: float | None = None
) -> tuple[Tensor, Tensor]:
    """
    Masks contact tokens of valid residues.

    If `mask_probability` is None, a masking ratio r ~ U(0, 1) is drawn per example
    and ceil(r * num_valid) tokens (at least one) are masked, so training covers all
    masking levels seen during sampling, from fully masked to almost fully unmasked.
    Otherwise every valid token is masked independently with `mask_probability`.
    """
    res_mask = res_mask.bool()

    if mask_probability is None:
        num_batch = tokens.shape[0]
        ratio = torch.rand(num_batch, device=tokens.device)
        num_valid = res_mask.sum(dim=-1)
        num_masked = torch.ceil(ratio * num_valid).clamp(min=1)

        # Mask the num_masked valid residues with the lowest random scores.
        scores = torch.rand_like(tokens.float()).masked_fill(~res_mask, float("inf"))
        ranks = scores.argsort(dim=-1).argsort(dim=-1)
        random_mask = (ranks < num_masked[:, None]) & res_mask
    else:
        random_mask = (torch.rand_like(tokens.float()) < mask_probability) & res_mask

    masked_tokens = tokens.clone()
    masked_tokens[random_mask] = MASK

    return masked_tokens, random_mask

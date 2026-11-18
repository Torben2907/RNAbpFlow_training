"""
Hybrid sampler: synchronizes the discrete contact tokens with the continuous flow.

The tokens are revealed one at a time (most confident first), and every reveal
restarts the continuous flow from the *same* initial noise:

    (x_init, c)   --flow-->  until a masked token is confident  --> reveal it: c'
    (x_init, c')  --flow-->  until a masked token is confident  --> reveal it: c''
    ...
    (x_init, c_full) --flow--> t = 1: the final structure x_end

Early in a trajectory the structure is a superposition of possible solutions; the
flow is followed only until the model is sure about one contact token. The final
trajectory, conditioned on all tokens, should then be close to a straight
("noising") path, i.e. easy to integrate.

Afterwards each sample gets a self-consistency score: some tokens are masked again,
and the model must recover them from the final structure,

    score = sum over remasked i of log P_i(k = c_i | x_end, c_remasked, t = 1).

Pseudocode (flow time t runs from min_t = noise to 1 = data in this code base):

    1  draw x_init per seed
    2  WHILE c contains masked tokens
    3      x(t) <- x_init, t <- min_t
    5      WHILE max_{i masked, k} P_i(k | x(t), c, t) < 1 - eps   (and t < 1)
    6          x(t + dt) <- Euler step towards x1_hat(x(t), c, t)
    7-8    reveal c_i <- argmax_k P_i(k | ...) for the most confident masked i
       integrate (x_init, c) from min_t to 1                       -> x_end
    9  c_re <- remask(c)
    10 score <- sum_{i remasked} log P_i(k = c_i | x_end, c_re, t = 1)
"""

import itertools
import math

import torch
from torch import Tensor

from astrafold.data.contacts import MASK
from astrafold.data.interpolant import Interpolant, draw_per_sample, seeded_generators


@torch.no_grad()
def sample_hybrid(
    interpolant: Interpolant,
    model: torch.nn.Module,
    onehot: Tensor,
    seeds: list[int],
    num_timesteps: int | None = None,
    confidence_tolerance: float = 0.1,
    tokens_per_round: int = 1,
    remask_fraction: float = 1.0,
) -> dict[str, Tensor]:
    """
    Generates one structure per seed for a single sequence with the hybrid sampler
    (see the module docstring).

    Samples run in one batch but are independent: each has its own flow time and
    restarts on its own, and all randomness comes from its seed. Samples whose
    tokens are all revealed leave the batch, so the batch shrinks over time.

    Params:
        interpolant          : provides the prior, the Euler step and the time grid
        model                : predicts x1 and the token logits (in eval mode)
        onehot               : [N, 4] one-hot sequence, on the device to sample on
        seeds                : one seed per sample (B = len(seeds))
        num_timesteps        : flow times of one trajectory (min_t ... 1); defaults
                               to `sampling.num_timesteps`
        confidence_tolerance : eps; a token is revealed once its predicted
                               probability is >= 1 - eps. A trajectory that reaches
                               t = 1 without such a token reveals its most confident
                               one anyway (a "forced" reveal).
        tokens_per_round     : most tokens revealed per trajectory. 1 follows the
                               pseudocode; larger values reveal every token above
                               1 - eps (at most this many) and save restarts.
        remask_fraction      : share of the tokens remasked for the score (at least
                               one token); the remasked positions are random.

    Returns:
        trans            : [B, N, 3] frame origins (in Angstrom) of x_end
        rotmats          : [B, N, 3, 3] frame orientations of x_end
        torsions         : [B, N, 9, 2] torsion angles as (sin, cos)
        contact_tokens   : [B, N] the generated contact tokens
        self_consistency : [B] the self-consistency score (<= 0, higher is better)
        remask           : [B, N] True where a token was remasked for the score
        reveal_t         : [B, N] flow time at which each token was revealed
        reveal_round     : [B, N] trajectory (0, 1, ...) in which it was revealed
        num_forced       : [B] reveals that reached t = 1 without confidence
        num_model_calls  : [B] forward passes spent on the sample
    """
    if num_timesteps is None:
        num_timesteps = interpolant._sample_cfg.num_timesteps
    if num_timesteps < 2:
        raise ValueError("Sampling needs at least 2 timesteps")
    if not 0 <= confidence_tolerance < 1:
        raise ValueError("confidence_tolerance must be in [0, 1)")
    if tokens_per_round < 1:
        raise ValueError("tokens_per_round must be at least 1")
    if not 0 < remask_fraction <= 1:
        raise ValueError("remask_fraction must be in (0, 1]")

    device = onehot.device
    interpolant.set_device(device)
    num_batch, num_res = len(seeds), onehot.shape[0]
    self_condition = interpolant._cfg.self_condition
    last_step = num_timesteps - 1
    ts = torch.linspace(interpolant._cfg.min_t, 1.0, num_timesteps, device=device)

    # Line 1: the initial noise, fixed for all trajectories of a sample.
    generators = seeded_generators(seeds)
    trans_init, rotmats_init = interpolant.sample_prior(generators, num_res, device)

    def run_model(idx, trans_t, rotmats_t, tokens, t, trans_sc) -> dict[str, Tensor]:
        """The model on the samples `idx`, each at its own flow time t [len(idx)]."""
        batch = {
            "res_mask": torch.ones(len(idx), num_res, device=device),
            "onehot": onehot[None].expand(len(idx), -1, -1),
            "trans_t": trans_t,
            "rotmats_t": rotmats_t,
            "contact_tokens": tokens,
            "t": t[:, None],
        }
        if self_condition:
            # Zeros, as at the start of a trajectory, are what the model uses when
            # there is no previous prediction.
            batch["trans_sc"] = trans_sc
        return model(batch)

    # Per-sample state of the current trajectory.
    tokens = torch.full((num_batch, num_res), MASK, device=device)
    trans_t, rotmats_t = trans_init.clone(), rotmats_init.clone()
    trans_sc = torch.zeros_like(trans_init)
    step = torch.zeros(num_batch, dtype=torch.long, device=device)  # index into ts

    reveal_t = torch.full((num_batch, num_res), float("nan"), device=device)
    reveal_round = torch.full_like(tokens, -1)
    rounds = torch.zeros_like(step)
    num_forced = torch.zeros_like(step)
    num_model_calls = torch.zeros_like(step)
    threshold = 1.0 - confidence_tolerance

    # Lines 2-8: reveal tokens, restarting the flow from x_init after every reveal.
    while True:
        active = (tokens == MASK).any(dim=-1).nonzero().squeeze(-1)
        if active.numel() == 0:
            break
        a_tokens, a_step = tokens[active], step[active]
        a_t = ts[a_step]
        out = run_model(
            active, trans_t[active], rotmats_t[active], a_tokens, a_t, trans_sc[active]
        )
        num_model_calls[active] += 1

        # Line 5: the confidence max_k P_i(k | x(t), c, t) of every masked token.
        probs = torch.softmax(out["token_logits"].float(), dim=-1)
        confidence, value = probs.max(dim=-1)  # [A, N]
        masked = a_tokens == MASK
        confidence = confidence.masked_fill(~masked, -1.0)
        best_confidence = confidence.max(dim=-1).values
        confident = best_confidence >= threshold
        reveal = confident | (a_step == last_step)  # [A]

        # Lines 7-8: reveal the most confident masked token (rank 0), plus further
        # tokens above the threshold if tokens_per_round > 1.
        rank = confidence.argsort(dim=-1, descending=True).argsort(dim=-1)
        chosen = (rank == 0) | ((rank < tokens_per_round) & (confidence >= threshold))
        chosen &= masked & reveal[:, None]

        tokens[active] = torch.where(chosen, value, a_tokens)
        reveal_t[active] = torch.where(chosen, a_t[:, None], reveal_t[active])
        reveal_round[active] = torch.where(
            chosen, rounds[active][:, None], reveal_round[active]
        )
        num_forced[active] += (reveal & ~confident).long()
        rounds[active] += reveal.long()

        # Line 6 for the samples that keep going: one Euler step.
        keep_going = ~reveal
        if keep_going.any():
            idx = active[keep_going]
            t_1 = a_t[keep_going]
            d_t = ts[a_step[keep_going] + 1] - t_1
            trans_t[idx], rotmats_t[idx] = interpolant.euler_step(
                d_t[:, None, None],
                t_1[:, None, None],
                {k: out[k][keep_going] for k in ("pred_trans", "pred_rotmats")},
                trans_t[idx],
                rotmats_t[idx],
            )
            trans_sc[idx] = out["pred_trans"][keep_going]
            step[idx] += 1

        # Line 3 for the samples that revealed a token: restart from x_init.
        idx = active[reveal]
        trans_t[idx], rotmats_t[idx] = trans_init[idx], rotmats_init[idx]
        trans_sc[idx] = 0.0
        step[idx] = 0

    # The final trajectory: (x_init, all tokens) from min_t to t = 1.
    all_idx = torch.arange(num_batch, device=device)
    trans_t, rotmats_t = trans_init.clone(), rotmats_init.clone()
    trans_sc = torch.zeros_like(trans_init)
    for t_1, t_2 in itertools.pairwise(ts):
        t = t_1.expand(num_batch)
        out = run_model(all_idx, trans_t, rotmats_t, tokens, t, trans_sc)
        trans_t, rotmats_t = interpolant.euler_step(
            t_2 - t_1, t_1, out, trans_t, rotmats_t
        )
        trans_sc = out["pred_trans"]
    # As in `Interpolant.sample`: x_end is the model's prediction at t = 1.
    t_end = ts[-1].expand(num_batch)
    final = run_model(all_idx, trans_t, rotmats_t, tokens, t_end, trans_sc)
    num_model_calls += num_timesteps

    # Lines 9-10: remask a random subset and score the recovery from x_end.
    num_remask = max(1, math.ceil(remask_fraction * num_res))
    remask_noise = draw_per_sample(
        generators, lambda g: torch.rand(num_res, generator=g), device
    )
    remask = remask_noise.argsort(dim=-1).argsort(dim=-1) < num_remask
    tokens_re = torch.where(remask, torch.full_like(tokens, MASK), tokens)
    scored = run_model(
        all_idx,
        final["pred_trans"],
        final["pred_rotmats"],
        tokens_re,
        t_end,
        final["pred_trans"],
    )
    num_model_calls += 1
    log_probs = torch.log_softmax(scored["token_logits"].float(), dim=-1)
    token_log_probs = log_probs.gather(-1, tokens[..., None]).squeeze(-1)
    self_consistency = (token_log_probs * remask).sum(dim=-1)

    return {
        "trans": final["pred_trans"],
        "rotmats": final["pred_rotmats"],
        "torsions": final["pred_torsions"].reshape(num_batch, num_res, -1, 2),
        "contact_tokens": tokens,
        "self_consistency": self_consistency,
        "remask": remask,
        "reveal_t": reveal_t,
        "reveal_round": reveal_round,
        "num_forced": num_forced,
        "num_model_calls": num_model_calls,
    }

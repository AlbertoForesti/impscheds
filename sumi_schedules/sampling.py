"""Sumi's confidence sampler with a *scheduled* per-step token budget.

This is ``sumi_eval.sampling.uniform_diffusion_sample`` restricted to the
``adaptive`` sampler, with exactly one change: ``tokens_per_step`` becomes a
per-step sequence ``k_j`` instead of a constant. Every piece of the actual
denoising -- the canvas init, the ``[EOS][BOS]`` anchoring, the ``denoise_end``
tail-freeze, the confidence heuristic itself -- is the upstream function,
imported and called unmodified, so nothing about *which* positions get committed
changes; only *how many* per step.

Passing a constant budget reproduces the stock sampler exactly, including its RNG
consumption (at ``temperature=0`` the commit is an argmax and draws no randomness
at all, so only the prior-random canvas init consumes the generator).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import torch

# Upstream, unmodified. If these move, we want a hard ImportError, not a silent
# divergence from the sampler we are claiming to reuse.
from sumi_eval.sampling import _adaptive_step, _compute_logits, _init_z

__all__ = ["uniform_diffusion_sample_scheduled"]


@torch.no_grad()
def uniform_diffusion_sample_scheduled(
    model,
    tokenizer,
    prompts: Optional[List[str]],
    seq_length: int,
    vocab_size: int,
    token_budgets: Sequence[int],
    seed: int = 0,
    device: Union[torch.device, str] = "cuda",
    frozen: Optional[List] = None,
    temperature: float = 0.0,
    denoise_end: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, List[str]]:
    """Denoise with ``len(token_budgets)`` steps, committing ``token_budgets[j]``
    positions at step ``j``.

    Arguments mirror :func:`sumi_eval.sampling.uniform_diffusion_sample`; the
    ``num_steps`` / ``schedule`` / ``min_log_snr`` / ``max_log_snr`` knobs are gone
    because the adaptive sampler never read them (the SNR grid is unused -- see that
    function's line 390) and the step count is now ``len(token_budgets)``.

    Returns ``(z, texts)`` exactly as upstream does.
    """
    budgets = [int(k) for k in token_budgets]
    if not budgets:
        raise ValueError("token_budgets must be non-empty")
    if min(budgets) < 0:
        raise ValueError(f"token_budgets must be non-negative, got min {min(budgets)}")

    device = torch.device(device)

    if prompts is None or len(prompts) == 0:
        batch_size = 1
        prompt_token_ids: Optional[List[List[int]]] = None
    else:
        batch_size = len(prompts)
        prompt_token_ids = [tokenizer.tokenize(p) for p in prompts]
        max_prompt_len = max((len(ids) for ids in prompt_token_ids), default=0)
        if max_prompt_len >= seq_length:
            raise ValueError(
                f"longest prompt has {max_prompt_len} tokens, "
                f"must be < seq_length {seq_length}"
            )

    generator = torch.Generator(device=device).manual_seed(seed)
    z, noise_mask = _init_z(
        prompt_token_ids, seq_length, vocab_size, batch_size, device, generator,
        frozen=frozen, denoise_end=denoise_end,
    )
    position_ids = torch.arange(seq_length, device=device).unsqueeze(0).expand(batch_size, -1)

    was_training = model.training
    model.eval()
    try:
        for k in budgets:
            logits = _compute_logits(model, z, position_ids, vocab_size)
            if k == 0:
                # A schedule may prescribe a no-op step (only reachable with
                # min_per_step=0); burn the NFE but leave the canvas alone.
                z_new = z
            else:
                z_new, _ = _adaptive_step(
                    z, logits, noise_mask, k, temperature, generator
                )
            z = torch.where(noise_mask, z_new, z)
    finally:
        if was_training:
            model.train()

    texts = [tokenizer.detokenize(z[i].tolist()) for i in range(batch_size)]
    return z, texts

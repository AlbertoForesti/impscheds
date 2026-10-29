"""``lm-eval`` plugin: Sumi's confidence sampler driven by a timestep schedule.

Registers ``--model sumi_sched``. It is a thin subclass of the upstream
``sumi_eval.model.SumiDiffusionLM``: prompt preparation, the ``[EOS][BOS]`` anchor,
answer extraction and the whole ``loglikelihood`` path are inherited untouched. The
only override is :meth:`SumiScheduledLM._generate_batch`, which turns the schedule
into a per-step ``tokens_per_step`` budget (see :mod:`sumi_schedules.schedules`) and
hands it to :func:`sumi_schedules.sampling.uniform_diffusion_sample_scheduled`.

Extra ``--model_args``::

    sampling_schedule_name   uniform | eds | wds
    sampling_schedule_path   schedule file written by sumi_schedules.build_schedules (not needed for uniform)
    schedule_eps             eps of the uniform baseline grid (default 1e-3)
    min_tokens_per_step      floor on k_j (default 1; 0 permits no-op steps)

Note on the ``NFE == max_new_tokens`` cell: with the default floor of 1 there is
exactly one feasible budget vector (all ones), so every schedule collapses to the
stock ``tokens_per_step=1`` sampler there. That is a property of the constraint, not
a bug -- it is the shared anchor point of the NFE curves.
"""

import logging

from lm_eval.api.registry import register_model

from sumi_eval.model import SumiDiffusionLM

from sumi_schedules.sampling import uniform_diffusion_sample_scheduled
from sumi_schedules.schedules import SCHEDULE_NAMES, resolve_token_budgets

logger = logging.getLogger(__name__)

__all__ = ["SumiScheduledLM"]


@register_model("sumi_sched")
class SumiScheduledLM(SumiDiffusionLM):
    """Sumi with an adaptive, schedule-derived ``tokens_per_step``."""

    def __init__(
        self,
        *args,
        sampling_schedule_name="uniform",
        sampling_schedule_path="",
        schedule_eps=1e-3,
        min_tokens_per_step=1,
        **kwargs,
    ):
        # The schedule is consumed by the confidence sampler with a greedy commit;
        # the parent's ancestral-oriented defaults (sampler="ancestral",
        # gen_temperature=1.0) would be wrong for every run.
        kwargs.setdefault("sampler", "adaptive")
        kwargs.setdefault("gen_temperature", 0.0)
        kwargs.setdefault("fill_mode", "anchor")
        super().__init__(*args, **kwargs)

        if self.gen_sampler != "adaptive":
            raise ValueError(
                "sumi_sched schedules a per-step token budget, which only the "
                f"confidence ('adaptive') sampler consumes; got {self.gen_sampler!r}"
            )
        if sampling_schedule_name not in SCHEDULE_NAMES:
            raise ValueError(
                f"sampling_schedule_name must be one of {SCHEDULE_NAMES}, "
                f"got {sampling_schedule_name!r}"
            )

        self.sampling_schedule_name = sampling_schedule_name
        self.sampling_schedule_path = sampling_schedule_path
        self.schedule_eps = float(schedule_eps)
        self.min_tokens_per_step = int(min_tokens_per_step)

        self.token_budgets = resolve_token_budgets(
            sampling_schedule_name,
            sampling_schedule_path,
            nfe=self.num_denoising_steps,
            num_tokens=self.max_new_tokens,
            eps=self.schedule_eps,
            min_per_step=self.min_tokens_per_step,
        )
        logger.info(
            "schedule=%s nfe=%d L=%d | budgets sum=%d min=%d max=%d | %s",
            sampling_schedule_name, self.num_denoising_steps, self.max_new_tokens,
            sum(self.token_budgets), min(self.token_budgets),
            max(self.token_budgets), self.token_budgets,
        )

    # ------------------------------------------------------------- generation
    def _generate_batch(self, chunk, ceiling):
        """Upstream ``SumiDiffusionLM._generate_batch`` with the scheduled sampler.

        The prompt/anchor/tail bookkeeping below is the parent's, kept verbatim so
        the canvas layout is bit-for-bit the one the stock harness builds.
        """
        preps = [self._prep_prompt(r, ceiling) for r in chunk]
        prompts = [p[0] for p in preps]
        prompt_lens = [p[1] for p in preps]
        canvas = self.canvas_length if self.canvas_length is not None else preps[0][2]
        frozen = [p[3] for p in preps] if self.fill_mode == "anchor" else None
        if self.fill_mode == "anchor" and not self.gen_adaptive_free_tail:
            denoise_end = [p[3][0][0] + 2 for p in preps]
        elif self.fill_mode == "window":
            denoise_end = [p[5] for p in preps]
        else:
            denoise_end = None

        z, _ = uniform_diffusion_sample_scheduled(
            model=self.model,
            tokenizer=self.tokenizer,
            prompts=prompts,
            seq_length=canvas,
            vocab_size=self.vocab_size,
            token_budgets=self.token_budgets,
            seed=self.rng.randint(0, 2 ** 32 - 1),
            device=self._device,
            frozen=frozen,
            temperature=self.gen_temperature,
            denoise_end=denoise_end,
        )
        return [
            self._extract_text(
                z[i, prompt_lens[i]:preps[i][5]].tolist(),
                preps[i][4], self._task_name(chunk[i]),
            )
            for i in range(len(chunk))
        ]

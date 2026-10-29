"""Estimate Sumi's entropy-production / Wasserstein rates on MBPP-train and invert
them into EDS / WDS timestep schedules (Algorithm 1). The rate math lives in
:mod:`sumi_schedules.rates`.

Outputs (relative to the working directory)::

    outputs/rates/sumi/rates-mbpp-L<L>.pt          ep_times, ep_rates, wass_rates, dynamical_activity
    outputs/schedules/sumi/{eds,wds}-mbpp-L<L>.pt  one ascending time list per NFE = 1, 2, 4, ..., tgt_nfe
    outputs/timing/sumi/thermo-mbpp-L<L>.json

Estimation geometry
-------------------
The canvas is built exactly as ``sumi_eval/model.py`` builds it for generation, so
the rate is measured on the distribution the sampler actually traverses::

    [ MBPP prompt (frozen, real) ][ reference code, noised at t ][EOS][BOS][ prior-random, frozen ]
                                  |<------ EPR summed here ----->|

Each row uses its own natural code length ``L_row = min(len(code), L)`` -- so the
data window is real code end to end, with no filler -- and its EPR is rescaled to an
``L``-token window (``* L / L_row``) so rows of different lengths are commensurate
and match the ``L * sigma(t)`` activity term.

Example (about 3 GPU-hours on one H200)::

    PYTHONPATH=. third_party/sumi/.venv/bin/python -m sumi_schedules.build_schedules \
        --src-nfe 512 --num-samples 64 --max-new-tokens 256
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import scipy.integrate
import torch

from sumi_schedules import rates as R

MBPP_PROMPT = (
    "You are an expert Python programmer, and here is your task: {text} "
    "Your code should pass these tests:\n\n{t0}\n{t1}\n{t2}\n[BEGIN]\n"
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pretrained", default="tohoku-nlp/sumi-7b")
    p.add_argument("--split", default="train")
    p.add_argument("--src-nfe", type=int, default=512,
                   help="resolution of the t-grid the rates are estimated on")
    p.add_argument("--tgt-nfe", type=int, default=256,
                   help="largest NFE to emit a schedule for (powers of two up to it)")
    p.add_argument("--num-samples", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--eps", type=float, default=1e-3)
    p.add_argument("--canvas-length", type=int, default=2048)
    p.add_argument("--max-new-tokens", type=int, default=256,
                   help="L: the generation content budget the schedule is built for")
    p.add_argument("--min-code-tokens", type=int, default=16,
                   help="drop reference solutions shorter than this")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--checkpoint-every", type=int, default=16,
                   help="re-write the rate file every N grid points")
    p.add_argument("--overwrite-rates", action="store_true",
                   help="recompute the rates even if a (partial) rate file exists")
    return p.parse_args()


def output_path(args, kind, name, ext=".pt") -> str:
    """outputs/<kind>/sumi/<name>-mbpp-L<L><ext>, as in udm_schedules.tasks."""
    path = os.path.join(args.out_dir, kind, "sumi", f"{name}-mbpp-L{args.max_new_tokens}{ext}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


# ------------------------------------------------------------------ model / data
def load_model(args):
    import transformers
    from sumi_eval.config import SumiConfig
    from sumi_eval.shims import ModelForwardShim, TokenizerShim

    cfg = SumiConfig()
    dtype = getattr(torch, args.dtype)
    hf_model = transformers.AutoModelForMaskGeneration.from_pretrained(
        args.pretrained, trust_remote_code=True, dtype=dtype,
    ).to(args.device).eval()
    hf_tok = transformers.AutoTokenizer.from_pretrained(
        args.pretrained, trust_remote_code=True,
    )
    model = ModelForwardShim(hf_model, vocab_size=int(cfg.vocab_size))
    tokenizer = TokenizerShim(hf_tok, vocab_size=int(cfg.vocab_size))
    return model, tokenizer, cfg


def build_rows(args, tokenizer):
    """``[(prompt_ids, code_ids)]`` from MBPP-train, longest solutions first.

    Preferring long reference solutions keeps the estimation windows close to the
    ``L``-token generation budget the schedule is built for.
    """
    from datasets import load_dataset

    ds = load_dataset("google-research-datasets/mbpp", "full", split=args.split)
    reserve = 2  # [EOS][BOS]
    rows = []
    for ex in ds:
        tests = list(ex["test_list"]) + ["", "", ""]
        prompt = MBPP_PROMPT.format(text=ex["text"].strip(),
                                    t0=tests[0], t1=tests[1], t2=tests[2])
        prompt_ids = list(tokenizer.tokenize(prompt))
        # The reference solution is a continuation, so drop the tokenizer's BOS.
        code_ids = list(tokenizer.tokenize(ex["code"]))
        if code_ids and code_ids[0] == tokenizer.bos:
            code_ids = code_ids[1:]
        code_ids = code_ids[: args.max_new_tokens]
        if len(code_ids) < args.min_code_tokens:
            continue
        if len(prompt_ids) + len(code_ids) + reserve >= args.canvas_length:
            continue
        rows.append((prompt_ids, code_ids))

    rows.sort(key=lambda r: -len(r[1]))
    rows = rows[: args.num_samples]
    if len(rows) < args.num_samples:
        raise ValueError(
            f"only {len(rows)} usable MBPP-{args.split} rows, wanted {args.num_samples}"
        )
    return rows


def build_canvases(args, rows, tokenizer, vocab_size, generator, device):
    """Clean canvases + the per-row content window bounds.

    Returns ``(x0 [N, S], starts [N], ends [N])``. Everything outside
    ``[start, end)`` is either the frozen prompt, the ``[EOS][BOS]`` anchor, or the
    frozen prior-random tail -- none of which is ever re-noised.
    """
    n, canvas = len(rows), args.canvas_length
    x0 = torch.randint(0, vocab_size, (n, canvas), dtype=torch.long,
                       device=device, generator=generator)  # the prior-random tail
    starts, ends = [], []
    for i, (prompt_ids, code_ids) in enumerate(rows):
        plen, clen = len(prompt_ids), len(code_ids)
        x0[i, :plen] = torch.tensor(prompt_ids, device=device)
        x0[i, plen:plen + clen] = torch.tensor(code_ids, device=device)
        x0[i, plen + clen] = tokenizer.eos
        x0[i, plen + clen + 1] = tokenizer.bos
        starts.append(plen)
        ends.append(plen + clen)
    return x0, starts, ends


# ------------------------------------------------------------------- estimation
@torch.no_grad()
def rates_at(args, model, x0, starts, ends, t, vocab_size, generator):
    """Dataset-averaged EPR at ``t``, normalised to an ``L``-token window."""
    total_epr, count = 0.0, 0
    for lo in range(0, x0.shape[0], args.batch_size):
        hi = min(lo + args.batch_size, x0.shape[0])
        xb, sb, eb = x0[lo:hi], starts[lo:hi], ends[lo:hi]

        z_t = xb.clone()
        for i in range(xb.shape[0]):
            a, b = sb[i], eb[i]
            z_t[i, a:b] = R.add_noise(xb[i, a:b], t, vocab_size, args.eps, generator)

        position_ids = torch.arange(xb.shape[1], device=xb.device).unsqueeze(0).expand(
            xb.shape[0], -1)
        logits = model(input_ids=z_t, position_ids=position_ids,
                       attention_mask=None, labels=None)

        epr = R.entropy_production_rate_rows(
            logits, z_t, sb, eb, t, vocab_size, args.eps)
        widths = torch.tensor([b - a for a, b in zip(sb, eb)],
                              dtype=torch.float64, device=epr.device)
        epr = epr * (args.max_new_tokens / widths)  # per-L-token-window
        total_epr += float(epr.sum())
        count += epr.numel()
    return total_epr / count


# ---------------------------------------------------------------- schedule build
def invert(cum_curve, ep_times, tgt_nfe, eps):
    """Equal-progress inversion (eq. 7); see udm_schedules/progress.py."""
    total = float(cum_curve[-1])
    out = []
    for log_nfe in range(int(np.log2(tgt_nfe)) + 1):
        times = np.linspace(eps, 1, 2 ** log_nfe + 1, endpoint=True)
        mapped = np.interp(times * total, cum_curve, ep_times).tolist()
        assert not any(np.isnan(t) for t in mapped), f"NaN in schedule at NFE {2**log_nfe}"
        out.append(mapped)
    return out


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ep_times = np.linspace(args.eps, 1 - args.eps, args.src_nfe).tolist()
    path = output_path(args, "rates", "rates")

    # Resume from a (possibly partial) checkpoint written on the same grid.
    done = []
    if os.path.exists(path) and not args.overwrite_rates:
        cached = torch.load(path, map_location="cpu", weights_only=False)
        ct = np.asarray(cached.get("ep_times", []), dtype=np.float64)
        if ct.shape == (len(ep_times),) and np.allclose(ct, ep_times):
            done = [float(r) for r in cached.get("ep_rates", [])]
            print(f"Resuming from {path}: {len(done)}/{len(ep_times)} grid points done")
        else:
            print(f"{path} was computed on a different grid; recomputing.")

    entropy_duration = 0.0
    ep_rates = list(done)
    if len(done) != len(ep_times):
        model, tokenizer, cfg = load_model(args)
        vocab_size = int(cfg.vocab_size)
        generator = torch.Generator(device=args.device).manual_seed(args.seed)
        rows = build_rows(args, tokenizer)
        widths = [len(c) for _, c in rows]
        print(f"{len(rows)} MBPP-{args.split} rows | code tokens "
              f"min={min(widths)} median={int(np.median(widths))} max={max(widths)}")
        x0, starts, ends = build_canvases(
            args, rows, tokenizer, vocab_size, generator, torch.device(args.device))

        from tqdm import tqdm

        t0 = time.perf_counter()
        pbar = tqdm(ep_times[len(ep_rates):], desc="entropy rates",
                    initial=len(ep_rates), total=len(ep_times))
        for t in pbar:
            r = rates_at(args, model, x0, starts, ends, t, vocab_size, generator)
            assert np.isfinite(r), f"non-finite entropy rate at t={t}"
            ep_rates.append(float(r))
            pbar.set_postfix(t=f"{t:.4f}", rate=f"{r:.4g}")
            if len(ep_rates) % args.checkpoint_every == 0:
                torch.save({"ep_times": ep_times, "ep_rates": ep_rates}, path)
        entropy_duration = time.perf_counter() - t0
        del model
        torch.cuda.empty_cache()

    ep_rates = np.asarray(ep_rates, dtype=np.float64)
    ep_times_np = np.asarray(ep_times, dtype=np.float64)

    n_neg = int((ep_rates < 0).sum())
    if n_neg:
        print(f"WARNING: {n_neg}/{len(ep_rates)} grid points have EPR < 0 "
              f"(min {ep_rates.min():.4g}); clamping to 0 for the cost curves.")
    ep_rates = np.clip(ep_rates, 0.0, None)

    activity = R.dynamical_activity(
        ep_times_np, args.max_new_tokens, args.eps).numpy()
    wass_rates = R.wasserstein_rate(ep_rates, activity).numpy()

    cum_entropy = scipy.integrate.cumulative_trapezoid(ep_rates, ep_times_np, initial=0.0)
    cum_wass = scipy.integrate.cumulative_trapezoid(wass_rates, ep_times_np, initial=0.0)

    torch.save({
        "ep_times": ep_times,
        "ep_rates": ep_rates.tolist(),
        "wass_rates": wass_rates.tolist(),
        "dynamical_activity": activity.tolist(),
        "meta": {
            "pretrained": args.pretrained, "split": args.split, "src_nfe": args.src_nfe,
            "num_samples": args.num_samples, "eps": args.eps, "seed": args.seed,
            "max_new_tokens": args.max_new_tokens, "canvas_length": args.canvas_length,
        },
    }, path)
    print(f"Saved rates to {path}")

    for name, cum in (("eds", cum_entropy), ("wds", cum_wass)):
        sched = invert(cum, ep_times_np, args.tgt_nfe, args.eps)
        sched_path = output_path(args, "schedules", name)
        torch.save(sched, sched_path)
        print(f"  wrote {sched_path}  (NFE 1..{2 ** (len(sched) - 1)})")

    with open(output_path(args, "timing", "thermo", ext=".json"), "w") as f:
        json.dump({"entropy_rates_runtime_sec": entropy_duration}, f, indent=2)


if __name__ == "__main__":
    main()

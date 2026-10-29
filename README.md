# Improved sampling schedules for Discrete Diffusion Models

Code for the paper *Improved sampling schedules for Discrete Diffusion Models*.

The reverse process of a uniform discrete diffusion model is analyzed through stochastic
thermodynamics. Two training-free sampling schedules follow from it: each one places the
sampling steps so that every step covers an equal share of a cumulative "progress" quantity.

- **EDS** (Entropic Discrete Schedule) uses the entropy production rate H_tot(t).
- **WDS** (Wasserstein Discrete Schedule) uses the Wasserstein rate sqrt(A(t)/2 · H_tot(t)),
  where A(t) is the dynamical activity.

Both are compared with the Uniform schedule, Jump Your Steps (JYS, Park et al., 2024), and the
CosMap and Logit-Normal schedules (Esser et al., 2024). The experiments cover a 10-dimensional
binomial distribution, monophonic piano music (tauLDR) and the Sumi-7B diffusion language model.

## Repository layout

```
udm_schedules/       core library
  thermo.py          entropy production rate, dynamical activity, Wasserstein rate (eqs. 3-6)
  progress.py        cumulative-progress inversion -> EDS / WDS schedules (eq. 7, Algorithm 1)
  heuristics.py      CosMap, Logit-Normal
  jys.py             Jump Your Steps baseline
  oracle.py          analytic score of the binomial benchmark
  noise.py           uniform-graph diffusion with loglinear / linear noise schedules (binomial)
  tauldr.py          tauLDR uniform-rate CTMC and sequence transformer (piano)
  sampling.py        Euler / tau-leaping sampler driven by any schedule
  data.py, tasks.py  datasets, metrics and task setup
scripts/             command-line entry points
  build_jys.py  build_thermo.py  build_heuristic.py  evaluate.py
sumi_schedules/      Sumi-7B integration (lm-eval plugin, rate estimation on MBPP)
experiments/         one driver per experiment: binomial.sh, piano.sh, sumi.sh (+ setup_sumi.sh)
figures/             one script per figure / table of the paper, and make_all.sh
```

Every generated file goes to `outputs/`:
`schedules/`, `rates/`, `samples/`, `metrics/`, `timing/`, `logs/` and `figures/`.

## Installation

Binomial and piano experiments (Python >= 3.10, PyTorch >= 2.1):

```bash
uv venv && source .venv/bin/activate     # or any virtualenv
uv pip install -e .                      # or: pip install -e .
```

**Sumi-7B experiments.** These run in their own environment, built on the upstream Sumi
evaluation code at a pinned commit:

```bash
experiments/setup_sumi.sh   # clones tohoku-nlp/sumi into third_party/sumi and runs `uv sync`
```

The model weights (`tohoku-nlp/sumi-7b`) and the benchmark datasets are downloaded from the
Hugging Face Hub on first use.

**Piano checkpoint and data.** Download the `piano` model and dataset of
[tauLDR](https://github.com/andrew-cr/tauLDR) from the
[tauLDR Dropbox folder](https://www.dropbox.com/scl/fo/zmwsav82kgqtc0tzgpj3l/h?dl=0&rlkey=k6d2bp73k4ifavcg9ldjhgu0s)
and place the files as follows:

```
weights/piano-ctmc/{ckpt.pt, config.yaml}
datasets/piano/{train.npy, test.npy}
```

## Reproducing the paper

Run each experiment driver from the repository root:

```bash
experiments/binomial.sh                # Fig. 1, Table 2 (binomial rows), Fig. 4 binomial panel
experiments/piano.sh                   # Fig. 3, Table 2 (piano row), Fig. 4 piano panel
experiments/sumi.sh mbpp               # Fig. 2, Table 3, Fig. 4 Sumi panel (L = 256)
experiments/sumi.sh humaneval
experiments/sumi.sh gsm8k              # (L = 64)
figures/make_all.sh                    # figures -> outputs/figures/, LaTeX tables -> stdout
```

The drivers skip results that already exist, so they can be interrupted and restarted.

| Paper | Produced by | Plotted by |
|---|---|---|
| Fig. 1 (binomial, KL vs NFE) | `experiments/binomial.sh` | `figures/fig1_binomial.py` |
| Fig. 2 (Sumi-7B, MBPP / GSM8K / HumanEval) | `experiments/sumi.sh <task>` | `figures/fig2_sumi.py` |
| Fig. 3 (piano, outlier score / Hellinger) | `experiments/piano.sh` | `figures/fig3_piano.py` |
| Fig. 4 (H_tot(t) and Wasserstein rate) | rate files written by the three drivers | `figures/fig4_rates.py` |
| Table 2 (schedule computation time) | `experiments/binomial.sh`, `experiments/piano.sh` | `figures/table2_timing.py` |
| Table 3 (WDS vs Uniform on Sumi-7B) | `experiments/sumi.sh <task>` | `figures/table3_sumi.py` |

**Settings (Table 1 of the paper).**

| | Rates estimated from | Evaluation |
|---|---|---|
| Binomial | N = 1024 samples, N_t = 1024 timesteps | 16384 samples, NFE 2–256, 5 seeds |
| Piano | N = 64 training sequences, N_t = 1024 | full test set, NFE 2–64 |
| Sumi-7B | N = 64 MBPP-train solutions, N_t = 512 | full test sets, greedy confidence sampler, seed 42 |

**Hardware.**
- The binomial benchmark uses the exact score. It runs on a CPU (`DEVICE=cpu`), but a GPU
  is much faster.
- Piano needs one GPU.
- Sumi-7B needs one large GPU. Estimating the rates takes about 3 GPU-hours on an H200.

Every size can be overridden through environment variables, which is useful for a quick
smoke test:

```bash
SRC_NFE=64 NUM_SAMPLES=256 TGT_NFE=16 NFES="4 16" SEEDS=42 EVAL_SAMPLES=4096 experiments/binomial.sh
LIMIT=8 NFES="16" experiments/sumi.sh humaneval
```

The scripts can also be called directly, for example:

```bash
python scripts/build_thermo.py --task binomial --binomial_n 31 --noise_schedule linear --max_length 10
python scripts/evaluate.py --task binomial --binomial_n 31 --noise_schedule linear --max_length 10 \
    --schedule wds --nfe 8 --seed 42
```

## Credits

This code builds on:

- [Jump Your Steps](https://github.com/enkeejunior1/jump-your-steps) (Park et al., 2024):
  the code base, the JYS baseline and the uniform-graph SEDD scheduler.
- [SEDD](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion) (Lou et al., 2024).
- [tauLDR](https://github.com/andrew-cr/tauLDR) (Campbell et al., 2022): the piano model,
  checkpoint and data.
- [Sumi](https://github.com/tohoku-nlp/sumi) (Ye et al., 2026): the Sumi-7B model and its
  evaluation harness.

Files adapted from these projects say so in their headers.

## License

MIT, see [LICENSE](LICENSE). Files adapted from the projects above keep the terms of their
original authors.

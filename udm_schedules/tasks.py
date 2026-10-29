"""Task setup shared by the command-line scripts: the binomial benchmark
(analytic score) and piano generation (pretrained tauLDR checkpoint)."""
import os
import random
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch

from udm_schedules import noise, tauldr
from udm_schedules.data import BinomialDataset, PianoDataset, binomial_metric_fn, piano_metric_fn
from udm_schedules.oracle import BinomialUniformOracle

TASKS = ("binomial", "piano")
# all reverse processes start from t = 1 and stop at this time
SAMPLE_EPS = 1e-5


def add_task_args(parser):
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--noise_schedule", default="loglinear", choices=["loglinear", "linear"],
                        help="binomial only; the piano CTMC has a fixed uniform rate")
    parser.add_argument("--eps", type=float, default=1e-5,
                        help="noise-schedule eps; piano always uses 1e-2 (tauLDR checkpoint)")
    parser.add_argument("--binomial_n", type=int, default=31)
    parser.add_argument("--binomial_p", type=float, default=0.5)
    parser.add_argument("--max_length", type=int, default=None,
                        help="binomial: number of i.i.d. dimensions (default 1); piano: generated length")
    parser.add_argument("--fix_length", type=int, default=0,
                        help="piano: number of conditioning (prompt) tokens")
    parser.add_argument("--ckpt_dir", default="weights/piano-ctmc",
                        help="piano: directory with tauLDR's ckpt.pt and config.yaml")
    parser.add_argument("--data_dir", default="datasets/piano",
                        help="piano: directory with train.npy and test.npy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", default="outputs", help="root of all generated files")
    parser.add_argument("--tag", default="", help="experiment tag appended to every output file name")
    return parser


def configure(args):
    """Fill in the task-dependent fields of args (no randomness involved)."""
    if args.task == "binomial":
        args.parameterization = "sedd"
        args.max_length = args.max_length or 1
        args.length = args.max_length
        args.num_vocabs = args.binomial_n + 1
    elif args.task == "piano":
        args.parameterization = "ctmc"
        args.length = 256
        args.num_vocabs = 129
        args.eps = 1e-2
    if args.max_length is None:
        args.max_length = args.length
    args.tag = args.tag or default_tag(args)
    return args


def default_tag(args):
    if args.task == "binomial":
        return f"n{args.binomial_n}-{args.noise_schedule}-L{args.max_length}"
    return "piano"


def output_path(args, kind, name, ext=".pt"):
    """outputs/<kind>/<task>/<name>-<tag><ext>"""
    path = os.path.join(args.out_dir, kind, args.task, f"{name}-{args.tag}{ext}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


class Stopwatch:
    """Wall-clock timer that waits for pending CUDA work before reading the clock."""

    def __init__(self, device):
        self.cuda = str(device).startswith("cuda")
        self.start = self._now()

    def _now(self):
        if self.cuda:
            torch.cuda.synchronize()
        return time.perf_counter()

    def elapsed(self):
        return self._now() - self.start


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model(args):
    if args.task == "binomial":
        model = BinomialUniformOracle(num_vocabs=args.num_vocabs, p=args.binomial_p)
    elif args.task == "piano":
        import ml_collections
        import yaml
        with open(Path(args.ckpt_dir) / "config.yaml") as f:
            cfg = ml_collections.ConfigDict(yaml.safe_load(f))
        cfg.device = args.device
        model = tauldr.UniformRateSequenceTransformerEMA(cfg, args.device)
        loaded_state = torch.load(Path(args.ckpt_dir) / "ckpt.pt", map_location=args.device, weights_only=False)
        model.load_state_dict(remove_module_from_keys(loaded_state['model']))

    # NOTE: for the piano EMA model, freezing the parameters before eval()
    # makes EMA.move_shadow_params_to_model_params a no-op, so the raw
    # (non-EMA) weights are used. Kept as is: the reported piano results were
    # produced this way.
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model


def load_scheduler(args, model):
    if args.task == "binomial":
        return noise.EulerScheduler(args)
    return tauldr.EulerScheduler(args, model)


def load_dataset(args, split, num_samples):
    if args.task == "binomial":
        return BinomialDataset(num_samples=num_samples, num_vocabs=args.num_vocabs, p=args.binomial_p, length=args.length)
    return PianoDataset(args.data_dir, split, num_samples=num_samples, device=args.device)


def load_metric_fn(args):
    if args.task == "binomial":
        return partial(binomial_metric_fn, num_vocabs=args.num_vocabs, p=args.binomial_p)
    ds = PianoDataset(args.data_dir, "test", num_samples=args.num_samples, device=args.device)
    return partial(piano_metric_fn, ds=ds, fix_length=args.fix_length, num_vocabs=args.num_vocabs, device=args.device)


def remove_module_from_keys(state_dict):
    # keys of the form a.b.module.c.d (DistributedDataParallel) -> a.b.c.d
    return {key.replace('.module.', '.'): value for key, value in state_dict.items()}

"""``python -m sumi_schedules <lm-eval flags>`` -- the lm-eval CLI with both Sumi
plugins registered (lm-eval does not auto-discover entry points)."""


def main():
    import sumi_eval.model  # noqa: F401  -- registers --model sumi
    import sumi_schedules.model  # noqa: F401  -- registers --model sumi_sched
    from lm_eval.__main__ import cli_evaluate

    cli_evaluate()


if __name__ == "__main__":
    main()

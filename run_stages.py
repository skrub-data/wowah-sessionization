"""
Run the hyperparameter search stages in sequence, chaining each stage's winner
into the next one's `frozen` mapping.

    pixi run python run_stages.py --stages gaps structural

Each stage is a separate Optuna study, so an interrupted run resumes where it
stopped. `--timeout` bounds each stage in wall-clock seconds; a sleeping machine
burns it.
"""

import argparse
import json
import sys
import time

import main

# The feature-table choices Stage 1 settled with a margin larger than the noise
# of a 2% sample. The three that were inside the noise are re-tested by `gaps`.
STAGE1_ROBUST = {
    "lags": (1, 2, 3),
    "diff_lags": (1, 2),
    "location_features": True,
    "add_gini": False,
    "use_monthly_features": True,
}

# `gaps` is an exhaustive grid (3 x 3 x 2), so it needs exactly 18.
N_ITER = {"gaps": 18, "structural": 40, "capacity": 60}


def run_stage(stage, frozen, n_iter, timeout, load_fraction):
    study_name = f"wowah_churn_{stage}"
    print(f"\n=== stage {stage!r}: {n_iter} trials, study {study_name!r}")
    print(f"    frozen: {json.dumps(frozen, default=str)}")
    t0 = time.time()
    search, env = main.random_search(
        "train",
        load_fraction=load_fraction,
        stage=stage,
        n_iter=n_iter,
        study_name=study_name,
        timeout=timeout,
        frozen=frozen,
    )
    search.fit(env)
    elapsed = time.time() - t0
    print(f"    done in {elapsed / 60:.1f} min, best score {search.best_score_:.4f}")
    return study_name


def main_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["gaps", "structural"],
        choices=["gaps", "structural", "capacity"],
        help="stages to run, in order",
    )
    parser.add_argument("--fraction", type=float, default=0.1)
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="per-stage wall-clock bound in seconds",
    )
    args = parser.parse_args()

    frozen = dict(STAGE1_ROBUST)
    for stage in args.stages:
        study_name = run_stage(
            stage,
            frozen,
            N_ITER[stage],
            args.timeout,
            args.fraction,
        )
        # Carry this stage's winner into the next one.
        frozen, score = main.best_params_from_study(study_name, stage, frozen)
        print(f"    winner carried forward (score {score:.4f}):")
        for k, v in frozen.items():
            print(f"      {k} = {v!r}")

    print("\nfinal frozen mapping:")
    print(json.dumps(frozen, default=str, indent=2))


if __name__ == "__main__":
    sys.exit(main_cli())

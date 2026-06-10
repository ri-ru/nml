"""One launcher for the whole project — training, ensembling, and both final submissions.

Every subcommand just prints and runs the real underlying command (no magic), so what you see here is
exactly what the README documents. Set NML_DATA (or drop the dataset at ./data/nml-2026) first.

    python run.py split                      # regenerate the fixed 80/20 split manifests
    python run.py train v46 300              # train one ensemble member on the 80% split
    python run.py train v46 300 --full       # train one ensemble member on 100% of the data
    python run.py train-all --full           # train all 6 members (add nothing for the 80% split)
    python run.py fit-weights                # cache val preds, fit the 2-group ensemble weights
    python run.py submit                     # final submission 1: the 6-model ensemble CSV
    python run.py submit-single v46          # final submission 2: a single-model CSV (default v46)
    python run.py reproduce                  # the entire pipeline from scratch (GPU, hours)
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

# The 6 ensemble members (mirror demarche/ensemble/_config.py MEMBERS).
MEMBERS = ["v46", "v46_s3", "v46_s7", "v60_w8", "v60_w5_s1", "v60_w8_s2"]


def sh(*cmd):
    """Print a command (as you'd type it) and run it, streaming output. Abort on failure."""
    print(f"\n$ python {' '.join(cmd[1:])}\n", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    cmd, args = sys.argv[1], sys.argv[2:]

    if cmd == "split":
        sh(PY, "utils/make_split.py")

    elif cmd == "train":
        if not args:
            raise SystemExit("usage: python run.py train <label> [epochs] [--full]")
        sh(PY, "demarche/ensemble/train_member.py", *args)

    elif cmd == "train-all":
        full = "--full" in args
        epochs = next((a for a in args if a.isdigit()), "300")
        for label in MEMBERS:
            extra = ["--full"] if full else []
            sh(PY, "demarche/ensemble/train_member.py", label, epochs, *extra)

    elif cmd == "fit-weights":
        sh(PY, "demarche/ensemble/save_val_tta_preds.py")
        sh(PY, "demarche/ensemble/ball_player_ensemble.py")

    elif cmd == "submit":
        sh(PY, "final/build_ensemble_6model.py")

    elif cmd == "submit-single":
        sh(PY, "final/build_single_model.py", *args)  # optional [label], default v46

    elif cmd == "reproduce":
        sh(PY, "utils/make_split.py")
        for label in MEMBERS:
            sh(PY, "demarche/ensemble/train_member.py", label, "300")            # 80% (for weight fitting)
            sh(PY, "demarche/ensemble/train_member.py", label, "300", "--full")  # 100% (for submission)
        sh(PY, "demarche/ensemble/save_val_tta_preds.py")
        sh(PY, "demarche/ensemble/ball_player_ensemble.py")
        sh(PY, "final/build_ensemble_6model.py")
        sh(PY, "final/build_single_model.py", "v46")

    else:
        print(__doc__)
        raise SystemExit(f"unknown command: {cmd}")


if __name__ == "__main__":
    main()

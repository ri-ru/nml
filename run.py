"""Launcher for all of your reproductibility needs, dear graders! 
drop the dataset into the repo root to do this. (nml-2026)


Commends: 
    python run.py split                      # redoes the fixed 80/20 split  
    python run.py train v46 300              # train one ensemble member on 80% split
    python run.py train v46 300 --full       # train one ensemble member on 100% 
    python run.py train-all --full           # train all 6 members (add nothing for the 80% split)
    python run.py fit-weights                # cache val preds+ fit the 2-group ensemble weights
    python run.py submit                     # makes the final submission 1: the 6-model ensemble CSV
    python run.py submit-single v46          # makes the final submission 1: a single-model CSV (default v46, because this is our best model
    python run.py reproduce                  # the entire pipeline from scratch (GPU!!!)
"""


import os
import subprocess
import sys
from pathlib import Path





ROOT = Path(__file__).resolve().parent
PY = sys.executable

#6 ensemble members
MEMBERS = ["v46", "v46_s3", "v46_s7", "v60_w8", "v60_w5_s1", "v60_w8_s2"]


def _resolve_data_env():
    """Point NML_DATA at the dataset. 
    it should be put into the main repo, for it to be named nml-2026
    """
    env = os.environ.copy()
    if env.get("NML_DATA"):
        return env  # respect an explicit override
    for candidate in (ROOT / "nml-2026", ROOT / "data" / "nml-2026"):
        if (candidate / "train" / "train").exists():
            env["NML_DATA"] = str(candidate)
            break
    return env


ENV = _resolve_data_env()


def sh(*cmd):
    """print the commands, run it, abort on failure. """
    print(f"\n$ python {' '.join(cmd[1:])}\n", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True, env=ENV)


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

"""path config for the pipeline.  
Run  from inside demarche/ensemble/
"""
import os
from pathlib import Path

ENSEMBLE = Path(__file__).resolve().parent    
ROOT = ENSEMBLE.parent.parent  

ARCH_DIR = ROOT / "demarche"            # v46  
WEIGHTS = ROOT / "outputs" / "weights"  # existing weights 

 # label -> (ball_w, seed). ball_w == 1.0 is  generalist; > 1.0 is a ball specialist.
 
MEMBERS = {
    "v60_w8":    (8.0, 0),  # ball specialists
    "v60_w5_s1": (5.0, 1),
    "v60_w8_s2": (8.0, 2),
    "v46_s3":    (1.0, 3), # player generalists
    "v46":       (1.0, 0),
    "v46_s7":    (1.0, 7),
}
VAL_DIR = ENSEMBLE / "val_preds"  # cached per-model val TTA tensors (gitignored)! must be redone. 
SOL_DIR = ROOT / "outputs" / "solutions"  
for _d in (VAL_DIR, SOL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

 
TRAIN_TXT = ROOT / "train_files.txt"
VAL_TXT = ROOT / "val_files.txt"

DATA = Path(os.environ.get("NML_DATA") or ROOT / "data" / "nml-2026")
if not (DATA / "train" / "train").exists():
    raise SystemExit(
        f"Dataset not found at {DATA}. Set NML_DATA to the dataset root (folder with "
        "train/train/*.pt and test/test/*.pt) or place it at ./data/nml-2026."
    )
TRAIN = str(DATA / "train" / "train")
TEST = str(DATA / "test" / "test")

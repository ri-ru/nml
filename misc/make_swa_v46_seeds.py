"""
Compute SWA-averaged weights for v46 seed runs.
For each seed S in {s1, s2, s3}: average the state_dicts at epochs 280/285/290/295/300
into a single model_v46_S_swa.pth.
"""
import torch
from pathlib import Path

WEIGHTS = Path(__file__).parent.parent / 'models' / 'weights'
SWA_EPOCHS = [280, 285, 290, 295, 300]
SEEDS = ['s1', 's2', 's3']


def average_state_dicts(paths):
    sds = [torch.load(p, weights_only=True, map_location='cpu') for p in paths]
    keys = sds[0].keys()
    avg = {}
    for k in keys:
        # Skip int buffers (e.g. position embeddings indices) — only average float tensors
        ts = [sd[k] for sd in sds]
        if ts[0].is_floating_point():
            avg[k] = torch.stack(ts).mean(dim=0)
        else:
            avg[k] = ts[0]  # keep first for non-float
    return avg


def main():
    for s in SEEDS:
        paths = []
        for ep in SWA_EPOCHS:
            if ep == 300:
                p = WEIGHTS / f'model_v46_{s}_300ep.pth'  # tagged from run_one.py
            else:
                p = WEIGHTS / f'model_v46_{s}_ep{ep}.pth'
            assert p.exists(), f'missing {p}'
            paths.append(p)

        print(f'{s}: averaging {len(paths)} checkpoints...')
        avg = average_state_dicts(paths)
        out = WEIGHTS / f'model_v46_{s}_swa.pth'
        torch.save(avg, out)
        print(f'  -> {out}')


if __name__ == '__main__':
    main()

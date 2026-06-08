"""
Compute SWA-averaged weights for any model that has all 5 epoch snapshots
(280/285/290/295/300) on disk. Skips models that don't.

Targets v46 seeds (s1..s8) and v58 family (base, s1..s3).
"""
import torch
from pathlib import Path

WEIGHTS = Path(__file__).parent.parent / 'models' / 'weights'
SWA_EPOCHS = [280, 285, 290, 295, 300]

# (model_prefix, swa_suffix) -> resolves to model_<prefix>_swa.pth
TARGETS = [f'v46_s{i}' for i in range(1, 9)] + ['v58'] + [f'v58_s{i}' for i in range(1, 4)] + ['v60']


def average_state_dicts(paths):
    sds = [torch.load(p, weights_only=True, map_location='cpu') for p in paths]
    keys = sds[0].keys()
    avg = {}
    for k in keys:
        ts = [sd[k] for sd in sds]
        if ts[0].is_floating_point():
            avg[k] = torch.stack(ts).mean(dim=0)
        else:
            avg[k] = ts[0]
    return avg


def main():
    for prefix in TARGETS:
        out = WEIGHTS / f'model_{prefix}_swa.pth'
        if out.exists():
            print(f'{prefix}: swa already exists, skipping')
            continue
        paths = []
        missing = False
        for ep in SWA_EPOCHS:
            if ep == 300:
                p = WEIGHTS / f'model_{prefix}_300ep.pth'
            else:
                p = WEIGHTS / f'model_{prefix}_ep{ep}.pth'
            if not p.exists():
                print(f'{prefix}: missing {p.name} — skip')
                missing = True
                break
            paths.append(p)
        if missing:
            continue
        print(f'{prefix}: averaging {len(paths)} checkpoints...')
        avg = average_state_dicts(paths)
        torch.save(avg, out)
        print(f'  -> {out.name}')


if __name__ == '__main__':
    main()

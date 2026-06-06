"""v72_w8: relational message-passing GNN (NBAv72Model) + ball-weighted MoG NLL (ball entity 8x).
Combines the GNN interaction with the validated ball-weighting lever; intended as a diverse
ball-specialist for the per-entity ensemble.
"""
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
import nba_v46
from nba_v46 import NBASampler, MultiStepMSE, WEIGHTS_DIR
from nba_v72 import NBAv72Model
from nba_v60_base import BallWeightedMoGNLLLoss


class Trainer(nba_v46.Trainer):
    def train(self):
        torch.manual_seed(0)
        random.seed(0)
        generator = torch.Generator('cpu').manual_seed(0)
        train_sampler = NBASampler(self.batch_size, self.train_dataset.max_start, shuffle=True)
        val_sampler = NBASampler(self.batch_size, self.val_dataset.max_start, shuffle=False)
        train_dl = DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=train_sampler, generator=generator)
        val_dl = DataLoader(self.val_dataset, batch_size=self.batch_size, sampler=val_sampler, generator=generator)

        model = NBAv72Model().to(self.device)
        train_loss_fn = BallWeightedMoGNLLLoss(8.0)
        val_loss_fn = MultiStepMSE()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        for epoch in range(self.epochs):
            print(f'Epoch {epoch + 1}/{self.epochs}')
            model.train()
            self._run_epoch(train_dl, train_loss_fn, model, optimizer, 'train')
            model.train(False)
            with torch.no_grad():
                self._run_epoch(val_dl, val_loss_fn, model, None, 'val')
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)
            scheduler.step()
        torch.save(model.state_dict(), WEIGHTS_DIR / 'model_v72_w8.pth')
        self.model = model

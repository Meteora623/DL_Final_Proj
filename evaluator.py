# evaluator.py
from typing import NamedTuple, List, Any, Optional, Dict
from itertools import chain
from dataclasses import dataclass
import os
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import numpy as np
from matplotlib import pyplot as plt

from models import Prober
from configs import ConfigBase

from dataset import WallDataset
from normalizer import Normalizer

@dataclass
class ProbingConfig(ConfigBase):
    probe_targets: str = "locations"
    lr: float = 1e-3
    epochs: int = 20
    schedule: str = "cosine"  # Simplified scheduler
    sample_timesteps: int = 30
    prober_arch: str = "256-256"

def location_losses(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    assert pred.shape == target.shape
    mse = (pred - target).pow(2).mean(dim=0)
    return mse

class ProbingEvaluator:
    def __init__(
        self,
        device: str,
        model: torch.nn.Module,
        probe_train_ds,
        probe_val_ds: dict,
        config: ProbingConfig = ProbingConfig(),
        quick_debug: bool = False,
    ):
        self.device = device
        self.config = config
        self.model = model
        self.model.eval()
        self.quick_debug = quick_debug
        self.ds = probe_train_ds  # This is a DataLoader now
        self.val_ds = probe_val_ds
        self.normalizer = Normalizer()

    def train_pred_prober(self):
        repr_dim = self.model.repr_dim
        dataset = self.ds
        model = self.model
        config = self.config
        epochs = config.epochs

        if self.quick_debug:
            epochs = 1
        # Get the shape of the target locations
        for batch in dataset:
            target_shape = batch.locations.shape[2:]  # Assuming [B, T, 2]
            break

        prober = Prober(
            embedding=repr_dim,
            arch=config.prober_arch,
            output_shape=target_shape,
        ).to(self.device)

        optimizer_pred_prober = torch.optim.Adam(prober.parameters(), lr=config.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_pred_prober, T_max=epochs)

        prober.train()

        for epoch in tqdm(range(epochs), desc="Probe prediction epochs"):
            epoch_loss = 0.0
            for batch in tqdm(dataset, desc="Probe prediction step"):
                init_states = batch.states[:, 0:1]  # [B, 1, C, H, W]
                actions = batch.actions  # [B, T-1, 2]
                locations = batch.locations  # [B, T, 2]

                init_states = init_states.to(self.device)
                actions = actions.to(self.device)
                locations = locations.to(self.device)

                with torch.no_grad():
                    pred_encs = model(init_states, actions)  # [T, B, D]

                pred_encs = pred_encs.permute(1, 0, 2)  # [B, T, D]
                pred_encs = pred_encs.reshape(-1, pred_encs.shape[-1])  # [B*T, D]

                # Normalize locations
                locations_norm = self.normalizer.normalize_location(locations)  # [B, T, 2]
                locations_norm = locations_norm.reshape(-1, locations_norm.shape[-1])  # [B*T, 2]

                # Forward pass through prober
                pred_locs = prober(pred_encs)  # [B*T, 2]

                # Compute MSE loss
                loss = F.mse_loss(pred_locs, locations_norm)

                optimizer_pred_prober.zero_grad()
                loss.backward()
                optimizer_pred_prober.step()

                epoch_loss += loss.item()

                if (epoch * len(dataset) + 0) % 100 == 0:
                    print(f"normalized pred locations loss {loss.item()}")

                if self.quick_debug:
                    break

            avg_loss = epoch_loss / len(dataset)
            print(f"Epoch [{epoch+1}/{epochs}] - Average Probing Loss: {avg_loss:.4f}")
            scheduler.step()

        prober.eval()
        return prober

    @torch.no_grad()
    def evaluate_all(self, prober):
        avg_losses = {}
        for dataset_name, val_ds in self.val_ds.items():
            avg_losses[dataset_name] = self.evaluate_pred_prober(
                prober=prober,
                val_ds=val_ds,
                prefix=dataset_name,
            )
        return avg_losses

    @torch.no_grad()
    def evaluate_pred_prober(self, prober, val_ds, prefix=""):
        quick_debug = self.quick_debug
        config = self.config
        model = self.model
        probing_losses = []
        prober.eval()

        for idx, batch in enumerate(tqdm(val_ds, desc="Eval probe pred")):
            init_states = batch.states[:, 0:1]  # [B, 1, C, H, W]
            actions = batch.actions  # [B, T-1, 2]
            locations = batch.locations  # [B, T, 2]

            init_states = init_states.to(self.device)
            actions = actions.to(self.device)
            locations = locations.to(self.device)

            with torch.no_grad():
                pred_encs = model(init_states, actions)  # [T, B, D]

            pred_encs = pred_encs.permute(1, 0, 2)  # [B, T, D]
            pred_encs = pred_encs.reshape(-1, pred_encs.shape[-1])  # [B*T, D]

            # Normalize locations
            locations_norm = self.normalizer.normalize_location(locations)
            locations_norm = locations_norm.reshape(-1, locations_norm.shape[-1])  # [B*T, 2]

            # Forward pass through prober
            pred_locs = prober(pred_encs)  # [B*T, 2]

            # Compute MSE loss
            loss = F.mse_loss(pred_locs, locations_norm)
            probing_losses.append(loss.item())

            if quick_debug and idx > 2:
                break

        # Average loss
        average_eval_loss = np.mean(probing_losses)
        # Unnormalize the MSE
        average_eval_loss = self.normalizer.unnormalize_mse(average_eval_loss)
        return average_eval_loss

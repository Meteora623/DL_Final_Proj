import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os

from dataset import WallDataset
from models import JEPA_Model
from normalizer import Normalizer

class TrainConfig:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_path = '/scratch/DL24FA/train'
    batch_size = 64
    num_workers = 4
    lr = 1e-3
    epochs = 2  # Increased epochs for better learning
    repr_dim = 256
    action_dim = 2
    model_weights_path = 'model_weights.pth'

def vicreg_loss(pred_encs, target_encs, sim_weight=25.0, var_weight=25.0, cov_weight=1.0):
    # VICReg-like loss
    # pred_encs, target_encs: [T, B, D]
    T, B, D = pred_encs.shape
    pred = pred_encs.view(T*B, D)
    target = target_encs.view(T*B, D).detach()

    # Invariance term (mean squared error)
    invariance_loss = F.mse_loss(pred, target)

    # Variance term: ensure each dimension of embeddings in a batch have unit variance
    def var_term(x):
        std = torch.sqrt(x.var(dim=0) + 1e-4)
        return torch.mean(F.relu(1 - std))

    var_loss = var_term(pred) + var_term(target)

    # Covariance term: decorrelate the dimensions
    def cov_term(x):
        x_centered = x - x.mean(dim=0, keepdim=True)
        cov = (x_centered.T @ x_centered) / (x_centered.size(0)-1)
        off_diag = cov.flatten()[:-1].view(cov.size(0)-1, cov.size(1)+1)[:,1:].flatten()
        return (off_diag**2).mean()

    cov_loss = cov_term(pred) + cov_term(target)

    loss = sim_weight * invariance_loss + var_weight * var_loss + cov_weight * cov_loss
    return loss

def main():
    config = TrainConfig()

    train_dataset = WallDataset(
        data_path=config.data_path,
        probing=False,
        device='cpu',
        augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=False,
        num_workers=config.num_workers
    )

    model = JEPA_Model(
        repr_dim=config.repr_dim,
        action_dim=config.action_dim,
        device=config.device
    ).to(config.device)

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    normalizer = Normalizer()

    model.train()

    for epoch in range(config.epochs):
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            states = batch.states  # CPU
            actions = batch.actions # CPU
            states = states.to(config.device)
            actions = actions.to(config.device)

            optimizer.zero_grad()
            pred_encs = model(states, actions)  # [T, B, D]

            B, T_state, C, H, W = states.shape
            T = actions.shape[1] + 1
            target_encs_list = []
            with torch.no_grad():
                for t in range(T):
                    s_prime_t = model.target_encoder(states[:, t])
                    target_encs_list.append(s_prime_t)
            target_encs = torch.stack(target_encs_list, dim=0) # [T, B, D]

            loss = vicreg_loss(pred_encs, target_encs)
            loss.backward()
            optimizer.step()
            model.update_target_encoder(momentum=0.99)

            epoch_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{config.epochs}] Step [{batch_idx+1}/{len(train_loader)}]: Loss = {loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{config.epochs}] - Average Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), config.model_weights_path)
    print(f"Model weights saved to {config.model_weights_path}")

if __name__ == "__main__":
    main()

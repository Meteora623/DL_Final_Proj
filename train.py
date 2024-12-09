import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from dataset import WallDataset, create_wall_dataloader
from models import JEPA_Model
from normalizer import Normalizer

import os

class TrainConfig:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_path = '/scratch/DL24FA/train'
    batch_size = 64
    num_workers = 4
    lr = 1e-3
    epochs = 10
    repr_dim = 256
    action_dim = 2
    model_weights_path = 'model_weights.pth'

def compute_loss(pred_encs, target_encs):
    """
    Compute a BYOL-like loss:
    We use a cosine similarity-based loss.
    
    pred_encs: [T, B, D]
    target_encs: [T, B, D] (detached)
    """
    # Flatten T and B for convenience
    T, B, D = pred_encs.shape
    pred = pred_encs.view(T*B, D)
    target = target_encs.view(T*B, D)
    target = target.detach()  # Stop gradients on target

    # Cosine similarity loss
    # cos_sim in [-1,1], we want to maximize cos_sim (minimize 2 - 2*cos_sim)
    cos_sim = F.cosine_similarity(pred, target, dim=-1).mean()
    loss = 2 - 2 * cos_sim
    return loss

def main():
    config = TrainConfig()

    # Create the training dataset and dataloader
    # probing=False since we just need states and actions for training JEPA
    # augment=False here; you can set True if you want to apply image augmentation
    train_dataset = WallDataset(
        data_path=config.data_path,
        probing=False,
        device='cpu',    # Load data on CPU and then move to GPU in training loop
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

    # Initialize JEPA Model
    model = JEPA_Model(
        repr_dim=config.repr_dim,
        action_dim=config.action_dim,
        device=config.device
    ).to(config.device)

    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    # Normalizer (if you need it for debugging or special normalization)
    normalizer = Normalizer()

    model.train()

    for epoch in range(config.epochs):
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            # batch.states: [B, T, C, H, W]
            # batch.actions: [B, T-1, 2]
            # batch.locations: empty since probing=False

            # Move states and actions to device
            states = batch.states.to(config.device)   # [B, T, C, H, W]
            actions = batch.actions.to(config.device) # [B, T-1, 2]

            optimizer.zero_grad()

            # Forward pass
            pred_encs = model(states, actions)  # [T, B, D]

            # Get target embeddings using the target encoder
            # For each time step t, encode observation o_t using target encoder
            B, T_state, C, H, W = states.shape
            T = actions.shape[1] + 1
            target_encs_list = []
            with torch.no_grad():
                for t in range(T):
                    s_prime_t = model.target_encoder(states[:, t])  # [B, D]
                    target_encs_list.append(s_prime_t)
            target_encs = torch.stack(target_encs_list, dim=0)  # [T, B, D]

            # Compute loss
            loss = compute_loss(pred_encs, target_encs)

            # Backprop and optimize
            loss.backward()
            optimizer.step()

            # Update target encoder
            model.update_target_encoder(momentum=0.99)

            epoch_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{config.epochs}] Step [{batch_idx+1}/{len(train_loader)}]: Loss = {loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{config.epochs}] - Average Loss: {avg_loss:.4f}")

    # Save the trained model weights
    torch.save(model.state_dict(), config.model_weights_path)
    print(f"Model weights saved to {config.model_weights_path}")

if __name__ == "__main__":
    main()

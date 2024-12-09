from typing import NamedTuple
import torch
import numpy as np
import torchvision.transforms as transforms

class WallSample(NamedTuple):
    states: torch.Tensor
    locations: torch.Tensor
    actions: torch.Tensor

class WallDataset:
    def __init__(
        self,
        data_path,
        probing=False,
        device="cpu",
        augment=False,
    ):
        self.device = device
        self.augment = augment
        self.states = np.load(f"{data_path}/states.npy", mmap_mode="r")
        self.actions = np.load(f"{data_path}/actions.npy")

        if probing:
            self.locations = np.load(f"{data_path}/locations.npy")
        else:
            self.locations = None

        # Adjust these transforms as needed or remove them if causing issues
        self.augmentation_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ])

    def __len__(self):
        return len(self.states)

    def __getitem__(self, i):
        # Copy the array to avoid the non-writable warning
        state_np = self.states[i].copy()  # [T, C, H, W]
        action_np = self.actions[i].copy() # [T-1, 2]

        states = torch.from_numpy(state_np).float()  # CPU tensor
        actions = torch.from_numpy(action_np).float() # CPU tensor

        # Apply augmentation if enabled
        if self.augment:
            augmented_states = []
            for frame in states:
                frame = self.augmentation_transforms(frame)
                augmented_states.append(frame)
            states = torch.stack(augmented_states)

        # Move to device after augmentation
        states = states.to(self.device)
        actions = actions.to(self.device)

        if self.locations is not None:
            locations = torch.from_numpy(self.locations[i].copy()).float().to(self.device)
        else:
            locations = torch.empty(0, device=self.device)

        return WallSample(states=states, locations=locations, actions=actions)

def create_wall_dataloader(
    data_path,
    probing=False,
    device="cpu",
    batch_size=64,
    train=True,
    augment=False,
):
    ds = WallDataset(
        data_path=data_path,
        probing=probing,
        device=device,
        augment=augment,
    )

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train,
        drop_last=True,
        pin_memory=False,
    )

    return loader

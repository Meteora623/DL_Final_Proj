# dataset.py
from typing import NamedTuple
import torch
import numpy as np
import torchvision.transforms as transforms

class WallSample(NamedTuple):
    states: torch.Tensor
    locations: torch.Tensor
    actions: torch.Tensor
    states_view2: torch.Tensor  # Second augmented view

class WallDataset:
    def __init__(
        self,
        data_path,
        probing=False,
        device="cpu",
        augment=False,
    ):
        self.device = device
        self.probing = probing
        self.augment = augment
        self.states = np.load(f"{data_path}/states.npy", mmap_mode="r")
        self.actions = np.load(f"{data_path}/actions.npy")
        if probing:
            self.locations = np.load(f"{data_path}/locations.npy")
        else:
            self.locations = None

        # Define data augmentation transforms
        if self.augment:
            self.augmentation_transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                transforms.ToTensor(),
            ])
        else:
            self.augmentation_transforms = transforms.Compose([
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.states)

    def __getitem__(self, i):
        state_np = self.states[i].copy()  # [T, C, H, W]
        action_np = self.actions[i].copy()  # [T-1, action_dim]

        # Apply augmentation to each frame for view1
        states = []
        for frame in state_np:
            frame = self.augmentation_transforms(frame)
            states.append(frame)
        states = torch.stack(states)  # [T, C, H, W]

        # Apply a different augmentation for the second view
        states_view2 = []
        for frame in state_np:
            frame = self.augmentation_transforms(frame)
            states_view2.append(frame)
        states_view2 = torch.stack(states_view2)  # [T, C, H, W]

        actions = torch.from_numpy(action_np).float()

        states = states.to(self.device)
        states_view2 = states_view2.to(self.device)
        actions = actions.to(self.device)

        if self.locations is not None:
            locations = torch.from_numpy(self.locations[i].copy()).float().to(self.device)
        else:
            locations = torch.empty(0, device=self.device)

        return WallSample(states=states, locations=locations, actions=actions, states_view2=states_view2)

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
        pin_memory=True,
        num_workers=4,
    )

    return loader

# models.py
from typing import List
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

def build_mlp(layers_dims: List[int]):
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)

class ResNetEncoder(nn.Module):
    def __init__(self, repr_dim=256):
        super().__init__()
        # Use ResNet50 as the encoder
        self.encoder = models.resnet50(pretrained=False)
        self.encoder.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Modify for 2-channel input
        self.encoder.fc = nn.Identity()  # Remove the final classification layer
        self.projection = nn.Linear(2048, repr_dim)  # ResNet50 outputs 2048-dim features

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.encoder(x)  # [B, 2048]
        x = self.projection(x)  # [B, repr_dim]
        x = F.normalize(x, dim=1)
        return x

class Predictor(nn.Module):
    def __init__(self, repr_dim=256, action_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(repr_dim + action_dim, repr_dim)
        self.bn1 = nn.BatchNorm1d(repr_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(repr_dim, repr_dim)

    def forward(self, embedding, action):
        x = torch.cat([embedding, action], dim=1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = F.normalize(x, dim=1)
        return x

class JEPA_Model(nn.Module):
    def __init__(self, repr_dim=256, action_dim=2, device="cuda"):
        super().__init__()
        self.device = device
        self.repr_dim = repr_dim
        self.action_dim = action_dim

        self.encoder = ResNetEncoder(repr_dim=self.repr_dim)
        self.predictor = Predictor(repr_dim=self.repr_dim, action_dim=self.action_dim)

        self.target_encoder = ResNetEncoder(repr_dim=self.repr_dim)
        self._initialize_target_encoder()

    def _initialize_target_encoder(self):
        for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    def update_target_encoder(self, momentum=0.99):
        for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            param_k.data = param_k.data * momentum + param_q.data * (1.0 - momentum)

    def forward(self, states, actions):
        """
        states: [B, T, C, H, W]
        actions: [B, T-1, action_dim]
        """
        B, T_state, C, H, W = states.shape
        T = actions.shape[1] + 1  # Including initial state
        pred_encs = []

        # Initial state embedding
        s_t = self.encoder(states[:, 0].to(self.device))  # [B, D]
        pred_encs.append(s_t)

        for t in range(T - 1):
            a_t = actions[:, t].to(self.device)  # [B, action_dim]
            s_tilde = self.predictor(s_t, a_t)  # [B, D]
            pred_encs.append(s_tilde)
            s_t = s_tilde

        pred_encs = torch.stack(pred_encs, dim=0)  # [T, B, D]
        return pred_encs

class Prober(nn.Module):
    def __init__(self, embedding: int, arch: str, output_shape: List[int]):
        super().__init__()
        self.output_dim = int(np.prod(output_shape))
        embedding = int(embedding)
        arch_list = list(map(int, arch.split("-"))) if arch != "" else []
        arch_list = [int(a) for a in arch_list]
        f = [embedding] + arch_list + [self.output_dim]
        f = [int(x) for x in f]

        layers = []
        for i in range(len(f) - 2):
            layers.append(nn.Linear(f[i], f[i + 1]))
            layers.append(nn.ReLU(True))
        layers.append(nn.Linear(f[-2], f[-1]))
        self.prober = nn.Sequential(*layers)

    def forward(self, e):
        output = self.prober(e)
        return output

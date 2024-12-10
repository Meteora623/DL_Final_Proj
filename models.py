from typing import List
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

def build_mlp(layers_dims: List[int]):
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)


class TransformerEncoder(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.ReLU(inplace=True),
            nn.Linear(int(dim * mlp_ratio), dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, N, D]
        x = x + self.dropout(self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0])
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


class VisionTransformerEncoder(nn.Module):
    def __init__(self, repr_dim=256, img_size=64, patch_size=8, in_chans=2, embed_dim=256, depth=4, num_heads=4):
        super().__init__()
        # Calculate number of patches
        assert img_size % patch_size == 0, "Image size must be divisible by patch size."
        self.num_patches = (img_size // patch_size) ** 2
        patch_dim = in_chans * patch_size * patch_size

        # Patch embedding
        self.patch_size = patch_size
        self.img_size = img_size
        self.proj = nn.Linear(patch_dim, embed_dim)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

        # Transformer layers
        self.blocks = nn.ModuleList([
            TransformerEncoder(dim=embed_dim, num_heads=num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Final linear to repr_dim
        self.fc = nn.Linear(embed_dim, repr_dim)

        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape

        # Reshape into patches
        # (B, C, H, W) -> (B, num_patches, patch_dim)
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        # patches shape: [B, C, H/patch, W/patch, patch_size, patch_size]
        # rearrange patches into (B, N, patch_dim)
        patches = patches.contiguous().permute(0,2,3,1,4,5).reshape(B, self.num_patches, -1)

        x = self.proj(patches) # [B, N, embed_dim]

        cls_tokens = self.cls_token.expand(B, -1, -1) # [B, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1) # [B, N+1, embed_dim]

        x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        # Take CLS token
        cls = x[:, 0] # [B, embed_dim]

        cls = self.fc(cls) # [B, repr_dim]
        cls = F.normalize(cls, dim=1)
        return cls


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

        # Use VisionTransformerEncoder instead of the original CNN encoder
        self.encoder = VisionTransformerEncoder(repr_dim=self.repr_dim, img_size=64, patch_size=8, in_chans=2, embed_dim=256, depth=4, num_heads=4)
        self.predictor = Predictor(repr_dim=self.repr_dim, action_dim=self.action_dim)

        self.target_encoder = VisionTransformerEncoder(repr_dim=self.repr_dim, img_size=64, patch_size=8, in_chans=2, embed_dim=256, depth=4, num_heads=4)
        self._initialize_target_encoder()

    def _initialize_target_encoder(self):
        for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    def update_target_encoder(self, momentum=0.99):
        for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            param_k.data = param_k.data * momentum + param_q.data * (1.0 - momentum)

    def forward(self, states, actions):
        B, T_state, C, H, W = states.shape
        T = actions.shape[1] + 1
        pred_encs = []

        s_t = self.encoder(states[:, 0].to(self.device))
        pred_encs.append(s_t)

        for t in range(T - 1):
            a_t = actions[:, t].to(self.device)
            s_tilde = self.predictor(s_t, a_t)
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

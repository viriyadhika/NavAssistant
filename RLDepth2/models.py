import os
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical

import torchvision.models as tv_models
from PIL import Image
from cons import DEVICE, FEAT_DIM, NUM_ACTIONS

class RGBResNetEncoder(nn.Module):
    """
    Frozen ResNet18 backbone + projection to FEAT_DIM
    Input:  (B,3,H,W), unnormalized [0,1]
    Output: (B, FEAT_DIM)
    """
    def __init__(self, feat_dim: int = FEAT_DIM, device: str = DEVICE):
        super().__init__()
        resnet = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # (B,512,1,1)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        self.proj = nn.Linear(resnet.fc.in_features, feat_dim)
        self.feat_dim = feat_dim
        self.device = device
        self.to(device)

        # ImageNet normalization constants
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,3,H,W), values in [0,1]
        """
        x = x.to(self.device)
        x = (x - self.mean) / self.std
        with torch.no_grad():
            f = self.backbone(x)  # (B,512,1,1)
            f = f.flatten(1)      # (B,512)
        f = self.proj(f)          # (B,feat_dim)
        return f


class DepthEncoder(nn.Module):
    """
    Trainable depth encoder.
    Input:  (B,1,H,W), assumed roughly in meters (scaled)
    Output: (B, FEAT_DIM)
    """
    def __init__(self, feat_dim: int = FEAT_DIM, device: str = DEVICE):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(256, feat_dim)
        self.feat_dim = feat_dim
        self.device = device
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,1,H,W)
        """
        x = x.to(self.device)
        f = self.cnn(x)   # (B,256,1,1)
        f = f.flatten(1)  # (B,256)
        f = self.proj(f)  # (B,feat_dim)
        return f


class FusedEncoder(nn.Module):
    """
    Encodes RGB + Depth into a single feature vector per frame.
    """
    def __init__(self, feat_dim: int = FEAT_DIM, device: str = DEVICE):
        super().__init__()
        self.rgb_encoder = RGBResNetEncoder(feat_dim, device)
        self.depth_encoder = DepthEncoder(feat_dim, device)

        self.fusion = nn.Sequential(
            nn.Linear(2 * feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
        )

        self.device = device
        self.feat_dim = feat_dim
        self.to(device)

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """
        rgb:   (B,3,H,W), [0,1]
        depth: (B,1,H,W), scaled e.g. depth/10.0
        returns: (B, feat_dim)
        """
        rgb_f = self.rgb_encoder(rgb)        # (B,D)
        depth_f = self.depth_encoder(depth)  # (B,D)
        fused = torch.cat([rgb_f, depth_f], dim=-1)
        fused = self.fusion(fused)           # (B,D)
        return fused
    

class ActorCritic(nn.Module):
    """
    Recurrent Actor-Critic with:
      - Fused RGB+Depth encoder
      - GRU over features
      - Policy and value heads
    """
    def __init__(
        self,
        feat_dim: int = FEAT_DIM,
        hidden_dim: int = 256,
        num_actions: int = NUM_ACTIONS,
        device: str = DEVICE,
    ):
        super().__init__()
        self.encoder = FusedEncoder(feat_dim, device)
        self.gru = nn.GRU(input_size=feat_dim, num_layers=8, hidden_size=hidden_dim, batch_first=True)

        self.policy_head = nn.Linear(hidden_dim, num_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        self.device = device
        self.to(device)

    def init_hidden(self, batch_size: int = 1) -> torch.Tensor:
        """
        Returns initial hidden state for GRU: (1, B, H)
        """
        return torch.zeros(1, batch_size, self.hidden_dim, device=self.device)

    def encode_obs(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """
        rgb:   (B,3,H,W)
        depth: (B,1,H,W)
        returns: (B,feat_dim)
        """
        return self.encoder(rgb, depth)

    def forward_sequence(
        self,
        feats: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        feats: (B,T,feat_dim)
        h0:    (1,B,H) or None
        returns:
            logits: (B,T,num_actions)
            values: (B,T)
            hN:     (1,B,H)
        """
        if h0 is None:
            h0 = self.init_hidden(batch_size=feats.size(0))

        feats = feats.to(self.device)
        out, hN = self.gru(feats, h0)            # (B,T,H), (1,B,H)

        logits = self.policy_head(out)          # (B,T,A)
        values = self.value_head(out).squeeze(-1)  # (B,T)

        return logits, values, hN

    def act(
        self,
        feat: torch.Tensor,
        h: torch.Tensor,
    ) -> Tuple[int, float, float, torch.Tensor]:
        """
        Single-step action.
        feat: (1,1,feat_dim)
        h:    (1,1,H)
        returns:
            action_idx, logp, value, h_new
        """
        logits, values, h_new = self.forward_sequence(feat, h)  # logits: (1,1,A)
        logits = logits[:, -1, :]             # (1,A)
        value  = values[:, -1]                # (1,)

        dist = Categorical(logits=logits)
        action = dist.sample()                # (1,)
        logp = dist.log_prob(action)          # (1,)

        return int(action.item()), float(logp.item()), float(value.item()), h_new

class TransformerActorCritic(nn.Module):
    """
    Transformer-based Actor-Critic with:
      - Fused RGB+Depth encoder
      - Causal Transformer over feature sequences
      - Policy and value heads
    API is compatible with your current GRU-based ActorCritic:
      - encode_obs(rgb, depth) -> (B, feat_dim)
      - forward_sequence(feats, h0) -> logits, values, hN
      - act(feat_seq, h) -> action, logp, value, h_new
    """
    def __init__(
        self,
        feat_dim: int = FEAT_DIM,
        hidden_dim: int = 256,           # kept for compatibility (used only in init_hidden)
        num_actions: int = NUM_ACTIONS,
        n_layers: int = 2,
        n_heads: int = 4,
        device: str = DEVICE,
    ):
        super().__init__()
        self.encoder = FusedEncoder(feat_dim, device)

        # Transformer encoder over feature sequences
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim,
            nhead=n_heads,
            dim_feedforward=4 * feat_dim,
            batch_first=True,
        )
        self.tr = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Policy & value heads work on per-timestep transformer outputs
        self.policy_head = nn.Linear(feat_dim, num_actions)
        self.value_head = nn.Linear(feat_dim, 1)

        # For PPOTrainer compatibility
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        self.device = device
        self.to(device)

    # -------------------------------------------------
    #  Hidden state API (kept for PPOTrainer)
    # -------------------------------------------------
    def init_hidden(self, batch_size: int = 1) -> torch.Tensor:
        """
        GRU-style API for compatibility.
        Transformer doesn't use h, but PPOTrainer expects a tensor.
        Shape: (1, B, H)
        """
        return torch.zeros(1, batch_size, self.hidden_dim, device=self.device)

    # -------------------------------------------------
    #  Encode RGB + Depth
    # -------------------------------------------------
    def encode_obs(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """
        rgb:   (B,3,H,W)
        depth: (B,1,H,W)
        returns: (B, feat_dim)
        """
        return self.encoder(rgb, depth)

    # -------------------------------------------------
    #  Forward full sequence (for PPO update)
    # -------------------------------------------------
    def forward_sequence(
        self,
        feats: torch.Tensor,                 # (B,T,feat_dim)
        h0: Optional[torch.Tensor] = None,   # ignored, kept for API
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        feats: (B,T,feat_dim)
        h0:    (1,B,H) or None (ignored)
        returns:
            logits: (B,T,num_actions)
            values: (B,T)
            hN:     (1,B,H) dummy (zeros)
        """
        B, T, D = feats.shape
        feats = feats.to(self.device)

        # Causal mask: each timestep can only attend to <= t
        # shape (T,T), with -inf above diagonal
        mask = torch.full((T, T), float('-inf'), device=self.device)
        mask = torch.triu(mask, diagonal=1)

        # Transformer encoder
        z = self.tr(feats, mask=mask)        # (B,T,D)

        logits = self.policy_head(z)         # (B,T,A)
        values = self.value_head(z).squeeze(-1)  # (B,T)

        # Dummy "next hidden state" for PPOTrainer compatibility
        hN = self.init_hidden(batch_size=B)

        return logits, values, hN

    # -------------------------------------------------
    #  Single-step action (for rollout)
    # -------------------------------------------------
    def act(
        self,
        feat_seq: torch.Tensor,   # (1,1,feat_dim) in your current trainer
        h: torch.Tensor,          # (1,1,H), ignored
    ) -> Tuple[int, float, float, torch.Tensor]:
        """
        Single-step action interface used in collect_rollout.
        feat_seq: (1,1,feat_dim)
        h:        (1,1,H) dummy (ignored)
        returns:
            action_idx, logp, value, h_new
        """
        logits, values, hN = self.forward_sequence(feat_seq, h0=None)  # logits: (1,1,A)
        logits = logits[:, -1, :]          # (1,A)
        value  = values[:, -1]             # (1,)

        dist = Categorical(logits=logits)
        action = dist.sample()             # (1,)
        logp = dist.log_prob(action)       # (1,)

        return int(action.item()), float(logp.item()), float(value.item()), hN
    
class SlidingWindowTransformerActorCritic(nn.Module):
    """
    Actor-Critic with:
      - Fused RGB+Depth encoder
      - Sliding-window causal Transformer (length W)
      - Policy and value heads
    Drop-in replacement for your current ActorCritic.
    """

    def __init__(
        self,
        feat_dim: int = FEAT_DIM,
        num_actions: int = NUM_ACTIONS,
        window: int = 32,
        n_layers: int = 2,
        n_heads: int = 4,
        device: str = DEVICE,
    ):
        super().__init__()
        self.encoder = FusedEncoder(feat_dim, device)

        self.window = window
        self.feat_dim = feat_dim
        self.device = device

        # Positional embeddings for window
        self.pos_embed = nn.Parameter(torch.zeros(1, window, feat_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer encoder
        layer = nn.TransformerEncoderLayer(
            d_model=feat_dim,
            nhead=n_heads,
            dim_feedforward=4 * feat_dim,
            batch_first=True,
        )
        self.tr = nn.TransformerEncoder(layer, num_layers=n_layers)

        # Heads
        self.policy_head = nn.Linear(feat_dim, num_actions)
        self.value_head = nn.Linear(feat_dim, 1)

        # For PPO API compatibility
        self.hidden_dim = feat_dim
        self.num_actions = num_actions
        self.to(device)

    # ------------------------------------------------------------------
    #  Hidden state API (dummy for compatibility)
    # ------------------------------------------------------------------
    def init_hidden(self, batch_size: int = 1):
        return torch.zeros(1, batch_size, self.hidden_dim, device=self.device)

    # ------------------------------------------------------------------
    #  Encode RGB+Depth to per-step feature vector
    # ------------------------------------------------------------------
    def encode_obs(self, rgb, depth):
        return self.encoder(rgb, depth)

    # ------------------------------------------------------------------
    #  Sliding window extraction
    # ------------------------------------------------------------------
    def _get_window(self, seq):
        """
        seq : (B, S, D)
        Returns: (B, W, D) containing last window frames (padded)
        """
        B, S, D = seq.shape
        W = self.window

        if S >= W:
            return seq[:, S - W:S, :]
        else:
            pad = W - S
            pad_vec = seq[:, :1, :].repeat(1, pad, 1)
            return torch.cat([pad_vec, seq], dim=1)

    # ------------------------------------------------------------------
    #  Full-sequence (for PPO training)
    # ------------------------------------------------------------------
    def forward_sequence(self, feats, h0=None):
        """
        feats: (B,T,D)
        returns:
            logits: (B,T,A)
            values: (B,T)
            hN:     dummy
        """
        B, T, D = feats.shape
        feats = feats.to(self.device)

        logits_list = []
        values_list = []

        for t in range(T):
            # window over [0..t]
            win = self._get_window(feats[:, :t+1, :])   # (B,W,D)

            # add positional embeddings
            win = win + self.pos_embed

            # causal mask
            mask = torch.full((self.window, self.window), float('-inf'), device=self.device)
            mask = torch.triu(mask, diagonal=1)

            # transformer
            z = self.tr(win, mask=mask)   # (B,W,D)

            # last token → output for this timestep
            token = z[:, -1, :]    # (B,D)

            logits_list.append(self.policy_head(token))
            values_list.append(self.value_head(token).squeeze(-1))

        logits = torch.stack(logits_list, dim=1)   # (B,T,A)
        values = torch.stack(values_list, dim=1)   # (B,T)

        hN = self.init_hidden(B)
        return logits, values, hN

    # ------------------------------------------------------------------
    #  Act in rollout (single-step)
    # ------------------------------------------------------------------
    def act(self, feat_seq, h):
        """
        feat_seq: (1,1,D) -- the PPO trainer gives only 1 step at a time
        """
        logits, values, hN = self.forward_sequence(feat_seq)

        logits = logits[:, -1, :]   # (1,A)
        values = values[:, -1]      # (1,)

        dist = Categorical(logits=logits)
        action = dist.sample()

        return (
            int(action.item()),
            float(dist.log_prob(action)),
            float(values.item()),
            hN
        )
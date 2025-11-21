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
import clip  # OpenAI CLIP

# cons.py (example – adjust to match your file)
NUM_ACTIONS   = 3                # e.g. ["MoveAhead", "RotateLeft", "RotateRight"]
FEAT_DIM      = 256
HIDDEN_DIM    = 256
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

LR            = 3e-4
GAMMA         = 0.99
GAE_LAMBDA    = 0.95
PPO_CLIP      = 0.2
VALUE_COEF    = 0.5
ENTROPY_COEF  = 0.05
MAX_GRAD_NORM = 0.5

EPISODE_STEPS = 512
TRAIN_EPOCHS  = 4

INTRINSIC_COEF  = 1.0    # scales CLIP reward
EXTRINSIC_COEF  = 1.0    # scales any external reward you add

ACTIONS = ["MoveAhead", "RotateLeft", "RotateRight"]

# A torchvision-like transform for RGB, e.g.:
from torchvision import transforms
transform = transforms.Compose([
    transforms.ToTensor(),   # (H,W,3) -> (3,H,W), [0,1]
    transforms.Resize((224, 224)),
])

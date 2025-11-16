import torch
from torch import nn
from torchvision import transforms
import torch.nn.functional as F
import clip
from collections import deque
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.models as models
from cons import IMG_SIZE, FEAT_DIM, EPISODE_STEPS, DEVICE

class FrozenResNetEncoder(nn.Module):
    """
    Frozen pretrained ResNet encoder that outputs (B, S, feat_dim)
    but processes frames in chunks to avoid GPU OOM.
    """
    def __init__(self, feat_dim: int = FEAT_DIM, device: str = DEVICE, chunk_size: int = 32):
        super().__init__()

        # Use small ResNet backbone
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Remove classification head
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # -> (B, 512, 1, 1)
        self.backbone.eval()  

        # Freeze everything
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.backbone_out_dim = resnet.fc.in_features  # 512
        self.proj = nn.Linear(self.backbone_out_dim, feat_dim)

        self.feat_dim = feat_dim
        self.chunk_size = chunk_size
        self.device = device
        self.to(device)

        # Precompute normalization tensors
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, S, C, H, W)
        returns: (B, S, feat_dim)
        """
        b, s, c, h, w = x.shape
        x = x.reshape(b * s, c, h, w)  # flatten sequence dimension

        feats_list = []

        # Process in chunks to avoid OOM
        for i in range(0, b * s, self.chunk_size):
            chunk = x[i:i + self.chunk_size].to(self.device)

            # Normalize for ResNet
            chunk = (chunk - self.mean) / self.std

            with torch.no_grad():
                f = self.backbone(chunk)    # (chunk, 512, 1, 1)
                f = f.flatten(1)            # (chunk, 512)

            # Project to feat_dim
            f = self.proj(f)                # (chunk, feat_dim)
            feats_list.append(f)

        # Concat chunk outputs
        feats = torch.cat(feats_list, dim=0)    # (B*S, feat_dim)

        # Restore original shape
        return feats.view(b, s, self.feat_dim)
    


class LSTMActor(nn.Module):
    def __init__(self, feat_dim: int, num_actions: int, truncation_len=32):
        super().__init__()
        self.num_layers = 8
        self.hidden_size = feat_dim
        self.truncation_len = truncation_len

        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=feat_dim,
            num_layers=self.num_layers,
            batch_first=True
        )
        self.head = nn.Linear(feat_dim, num_actions)

    def forward(self, X, action_seq, mask):
        """
        X: (B, S, D)
        returns:
            logits: (B*S, num_actions)
            (h, c): final hidden state
        """
        B, S, D = X.shape
        T = self.truncation_len

        # ---- initial hidden state ----
        h = torch.zeros(self.num_layers, B, D, device=X.device)
        c = torch.zeros(self.num_layers, B, D, device=X.device)
        outputs = []

        # ---- truncated BPTT loop ----
        for start in range(0, S, T):
            end = min(S, start + T)
            chunk = X[:, start:end, :]   # (B, T, D)

            out, (h, c) = self.lstm(chunk, (h, c))

            # detach hidden state so gradients don’t flow across chunks
            h = h.detach()
            c = c.detach()

            outputs.append(out)

        # ---- concatenate outputs ----
        outputs = torch.cat(outputs, dim=1)   # (B, S, D)

        logits = self.head(outputs.reshape(B * S, D))
        return logits


class LSTMCritic(nn.Module):
    def __init__(self, feat_dim: int, truncation_len=32):
        super().__init__()
        self.num_layers = 8
        self.hidden_size = feat_dim
        self.truncation_len = truncation_len

        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=feat_dim,
            num_layers=self.num_layers,
            batch_first=True
        )
        self.head = nn.Linear(feat_dim, 1)

    def forward(self, X, mask):
        """
        X: (B, S, D)
        returns:
            logits: (B*S, num_actions)
            (h, c): final hidden state
        """
        B, S, D = X.shape
        T = self.truncation_len

        # ---- initial hidden state ----
        h = torch.zeros(self.num_layers, B, D, device=X.device)
        c = torch.zeros(self.num_layers, B, D, device=X.device)
        outputs = []

        # ---- truncated BPTT loop ----
        for start in range(0, S, T):
            end = min(S, start + T)
            chunk = X[:, start:end, :]   # (B, T, D)

            out, (h, c) = self.lstm(chunk, (h, c))

            # detach hidden state so gradients don’t flow across chunks
            h = h.detach()
            c = c.detach()

            outputs.append(out)

        # ---- concatenate outputs ----
        outputs = torch.cat(outputs, dim=1)   # (B, S, D)

        logits = self.head(outputs.reshape(B * S, D))
        return logits

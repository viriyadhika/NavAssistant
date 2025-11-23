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
        return feats.view(b, s, self.feat_dim) # (B, S, feat_dim)
    
class SmallCNNEncoder(nn.Module):
    """
    A small CNN encoder that outputs (B, S, feat_dim).
    Fully trainable, chunk-friendly, and avoids dummy pass.
    """
    def __init__(self, feat_dim: int, IMG_SIZE: int = IMG_SIZE,
                 device: str = "cuda", chunk_size: int = 32):
        super().__init__()

        self.IMG_SIZE = IMG_SIZE
        self.chunk_size = chunk_size
        self.device = device
        self.feat_dim = feat_dim

        # Same conv layout as before
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),  # -> (32, H1, W1)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # -> (64, H2, W2)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # -> (64, H3, W3)
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute final feature dimension analytically
        self.cnn_out_dim = self._compute_cnn_output_dim(IMG_SIZE)

        self.proj = nn.Linear(self.cnn_out_dim, feat_dim)

        self.to(device)

    def _conv_out(self, size, kernel, stride, padding=0):
        """Compute output size of a conv layer."""
        return (size - kernel + 2*padding) // stride + 1

    def _compute_cnn_output_dim(self, size):
        """
        Compute output dimension of the final flattened CNN output.
        CNN:
        Conv(8x8, s4)
        Conv(4x4, s2)
        Conv(3x3, s1)
        """
        # Layer 1
        h1 = self._conv_out(size, kernel=8, stride=4)
        # Layer 2
        h2 = self._conv_out(h1, kernel=4, stride=2)
        # Layer 3
        h3 = self._conv_out(h2, kernel=3, stride=1)

        # Final channels = 64
        return 64 * h3 * h3

    def forward(self, x):
        """
        x: (B, S, C, H, W)
        returns: (B, S, feat_dim)
        """
        B, S, C, H, W = x.shape
        x = x.reshape(B * S, C, H, W)  # flatten sequence dim

        feats = []
        for i in range(0, B * S, self.chunk_size):
            chunk = x[i:i+self.chunk_size].to(self.device)
            f = self.cnn(chunk)      # (chunk, cnn_out_dim)
            f = self.proj(f)         # (chunk, feat_dim)
            feats.append(f)

        feats = torch.cat(feats, dim=0)
        return feats.view(B, S, self.feat_dim)

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

    def forward(self, X, actions_seq, mask):
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


class SlidingWindowTransformer(nn.Module):
    def __init__(self, feat_dim: int, num_output: int, window=32, n_layers=4, n_heads=8):
        super().__init__()
        self.window = window

        self.pos_embed = nn.Parameter(torch.zeros(1, window, feat_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=feat_dim,
            nhead=n_heads,
            dim_feedforward=4 * feat_dim,
            batch_first=True
        )
        self.tr = nn.TransformerEncoder(layer, num_layers=n_layers)

        self.num_output = num_output
        self.action_head = nn.Linear(feat_dim, num_output)

    def forward(self, X, action_seq, mask, chunk_size=64):
        """
        X: (B, S, D)
        """
        B, S, D = X.shape
        W = self.window

        # -------------------------------------------------
        # 1) Build sliding index matrix (S, W)
        # -------------------------------------------------
        idx = torch.arange(S, device=X.device).unsqueeze(1) - torch.arange(W, device=X.device)
        idx = idx.clamp(min=0)   # early timesteps repeat frame 0

        # this will accumulate outputs for all S windows
        outputs = []

        # -------------------------------------------------
        # Chunked sliding window processing
        # -------------------------------------------------
        for start in range(0, S, chunk_size):
            end = min(start + chunk_size, S)

            # (C, W)
            idx_chunk = idx[start:end]

            # -------------------------------------------------
            # 2) Gather windows: (B, C, W, D)
            # -------------------------------------------------
            windows = X[:, idx_chunk, :]  # broadcasting idx_chunk

            # -------------------------------------------------
            # 3) Add positional embedding
            # -------------------------------------------------
            windows = windows + self.pos_embed[:, :W, :]

            # -------------------------------------------------
            # 4) Flatten windows → (B*C, W, D)
            # -------------------------------------------------
            BC = B * (end - start)
            windows = windows.reshape(BC, W, D)

            # -------------------------------------------------
            # 5) Transformer on chunk
            # -------------------------------------------------
            out = self.tr(windows)  # (B*C, W, D)

            # last token only
            last = out[:, -1, :]    # (B*C, D)

            # reshape back → (B, C, D)
            last = last.reshape(B, end - start, D)

            outputs.append(last)

        # -------------------------------------------------
        # 6) Concatenate across all chunks → (B, S, D)
        # -------------------------------------------------
        final = torch.cat(outputs, dim=1)

        # -------------------------------------------------
        # 7) Action head → (B, S, num_output)
        # -------------------------------------------------
        logits = self.action_head(final)

        # follow your template behavior exactly
        return logits.view(B * S, self.num_output)

class SlidingWindowTransformerActor(SlidingWindowTransformer):
    def __init__(self, feat_dim, num_actions, window=32, n_layers=4, n_heads=8):
        super().__init__(feat_dim, num_actions, window, n_layers, n_heads)

    def forward(self, X, action_seq, mask):
        return super().forward(X, action_seq, mask)

class SlidingWindowTransformerCritic(SlidingWindowTransformer):
    def __init__(self, feat_dim, window=32, n_layers=4, n_heads=8):
        super().__init__(feat_dim, 1, window, n_layers, n_heads)

    def forward(self, X, mask):
        return super().forward(X, None, mask)
    

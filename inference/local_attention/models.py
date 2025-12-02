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

class CompleteFrozenResNetEncoder(nn.Module):
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
        for p in self.proj.parameters():
            p.requires_grad = False

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

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, B: int, S: int, device):
        """
        Returns (B, S, dim) sinusoidal positional encodings.
        """
        position = torch.arange(S, device=device).unsqueeze(1)              # (S, 1)
        div_term = torch.exp(
            torch.arange(0, self.dim, 2, device=device) *
            (-torch.log(torch.tensor(10000.0, device=device)) / self.dim)
        )                                                                    # (dim/2,)

        pe = torch.zeros(S, self.dim, device=device)                         # (S, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(0).expand(B, S, self.dim)

class LocalWindowTransformer(nn.Module):
    """
    Causal local-attention transformer:
      - Input:  X (B, S, D)
      - Each position attends only to the last `window` positions (including itself)
      - Output: logits (B*S, num_output)  -- matches your original API
    """
    def __init__(self, feat_dim: int, num_output: int,
                 window: int = 32, n_layers: int = 4, n_heads: int = 8):
        super().__init__()
        self.window = window
        self.feat_dim = feat_dim

        # Positional encoding (works for any S)
        self.pos_embed = SinusoidalPositionalEmbedding(feat_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=feat_dim,
            nhead=n_heads,
            dim_feedforward=4 * feat_dim,
            batch_first=True,      # so Transformer sees (B, S, D)
        )
        self.tr = nn.TransformerEncoder(layer, num_layers=n_layers)

        self.num_output = num_output
        self.action_head = nn.Linear(feat_dim, num_output)

    def build_local_causal_mask(self, S: int, device):
        """
        Returns a boolean mask of shape (S, S):
          True  => position is MASKED (disallowed)
          False => allowed
        Enforces:
          - causal (no attending to future positions)
          - local window of size `self.window`
        """
        idx = torch.arange(S, device=device)
        i = idx.unsqueeze(1)  # (S, 1) target positions
        j = idx.unsqueeze(0)  # (1, S) source positions

        # Disallow attention to:
        too_future = j > i                    # j > i  => future
        too_past  = (i - j) >= self.window    # i - j >= W  => older than W-1

        mask = too_future | too_past          # (S, S) boolean
        return mask

    def forward(self, X, action_seq=None, mask=None):
        """
        X: (B, S, D)
        action_seq, mask: kept for API compatibility (ignored here)
        Returns: (B*S, num_output)
        """
        B, S, D = X.shape
        device = X.device

        # Add positional embeddings
        pos = self.pos_embed(B, S, device)    # (B, S, D)
        X = X + pos

        # Local causal attention mask: (S, S)
        attn_mask = self.build_local_causal_mask(S, device)

        # Transformer over full sequence with local attention
        out = self.tr(X, mask=attn_mask)      # (B, S, D)

        # Project to actions
        logits = self.action_head(out)        # (B, S, num_output)

        return logits.view(B * S, self.num_output)

class SlidingWindowTransformerActor(LocalWindowTransformer):
    def __init__(self, feat_dim, num_actions, window=32, n_layers=4, n_heads=8):
        super().__init__(feat_dim, num_actions, window, n_layers, n_heads)

    def forward(self, X, action_seq, mask):
        return super().forward(X, action_seq, mask)

class SlidingWindowTransformerCritic(LocalWindowTransformer):
    def __init__(self, feat_dim, window=32, n_layers=4, n_heads=8):
        super().__init__(feat_dim, 1, window, n_layers, n_heads)

    def forward(self, X, mask):
        return super().forward(X, None, mask)
    

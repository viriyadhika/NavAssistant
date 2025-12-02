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
    def __init__(self, feat_dim: int = FEAT_DIM, device: str = DEVICE, chunk_size: int = 32, project_to_out_dim=True):
        super().__init__()

        # Use small ResNet backbone
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Remove classification head
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # -> (B, 512, 1, 1)
        self.backbone.eval()
        self.project_to_out_dim = project_to_out_dim

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
            if self.project_to_out_dim:
                f = self.proj(f)                # (chunk, feat_dim)
            
            feats_list.append(f)

        # Concat chunk outputs
        feats = torch.cat(feats_list, dim=0)    # (B*S, feat_dim)

        # Restore original shape
        _, d = feats.shape
        return feats.view(b, s, d)
    

class FrozenResNetPCAEncoder(nn.Module):
    def __init__(self, feat_dim, pca_matrix, device="cuda", chunk_size=32):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.backbone.eval()

        for p in self.backbone.parameters():
            p.requires_grad = False

        self.register_buffer("pca_matrix", pca_matrix)  # (feat_dim, 512)

        # imagenet normalization
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

        self.chunk_size = chunk_size
        self.feat_dim = feat_dim
        self.device = device

    def forward(self, x):
        B, S, C, H, W = x.shape
        x = x.reshape(B*S, C, H, W)

        feats = []
        for i in range(0, B*S, self.chunk_size):
            chunk = x[i:i+self.chunk_size]
            chunk = (chunk - self.mean) / self.std

            with torch.no_grad():
                f = self.backbone(chunk).flatten(1)  # (N, 512)

            # PCA projection
            f = f @ self.pca_matrix.T   # (N, feat_dim)
            feats.append(f)

        feats = torch.cat(feats, 0)
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


class SlidingWindowTransformerActor(nn.Module):
    def __init__(self, feat_dim: int, num_actions: int, window=32, n_layers=4, n_heads=8):
        """
        feat_dim : embedding dimension
        window   : sliding window length (e.g., 32)
        """
        super().__init__()
        self.window = window

        # positional embedding for window
        self.pos_embed = nn.Parameter(torch.zeros(1, window, feat_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # transformer encoder
        layer = nn.TransformerEncoderLayer(
            d_model=feat_dim,
            nhead=n_heads,
            dim_feedforward=4 * feat_dim,
            batch_first=True
        )
        self.tr = nn.TransformerEncoder(layer, num_layers=n_layers)

        # value head
        self.value_head = nn.Linear(feat_dim, num_actions)

    def _get_window(self, seq):
        """
        seq : (B, S, D)
        returns last `window` frames padded to full length
        """
        B, S, D = seq.shape
        W = self.window

        if S >= W:
            return seq[:, S-W:S, :]     # take last W
        else:
            pad = W - S
            # pad by repeating the first frame (or zeros)
            pad_block = seq[:, :1, :].repeat(1, pad, 1)
            return torch.cat([pad_block, seq], dim=1)
    
    def _transformer_chunked(self, x, chunk_size=128):
        """
        x: (N, W, D)
        returns: (N, W, D)
        """
        outputs = []
        N = x.size(0)

        for i in range(0, N, chunk_size):
            chunk = x[i:i+chunk_size]   # (chunk, W, D)
            out = self.tr(chunk)        # transformer on mini-batch
            outputs.append(out)

        return torch.cat(outputs, dim=0)

    def forward(self, X, actions_seq, mask):
        """
        X: (B, S, D)
        mask: ignored (kept for API compatibility)
        returns: (B*S, 1)
        """
        B, S, D = X.shape

        # ---- 1. Build sliding windows ----
        # For each t, we build a window of length W ending at t
        W = self.window

        # Create tensor to hold all windows
        # windows[t] = window ending at timestep t
        windows = []
        for t in range(S):
            seq_t = X[:, :t+1, :]      # (B, t+1, D)
            win_t = self._get_window(seq_t)  # (B, W, D)
            windows.append(win_t)

        # Stack → (B, S, W, D)
        windows = torch.stack(windows, dim=1)

        # reshape into (B*S, W, D)
        flat = windows.reshape(B*S, W, D)

        # ---- 2. Add positional encoding ----
        flat = flat + self.pos_embed

        # ---- 3. Transformer ----
        z = self._transformer_chunked(flat)           # (B*S, W, D)

        # ---- 4. Use last token only ----
        last_token = z[:, -1]       # (B*S, D)

        # ---- 5. Value head ----
        values = self.value_head(last_token) # (B*S, 1)

        return values


class SlidingWindowTransformerCritic(nn.Module):
    def __init__(self, feat_dim: int, window=32, n_layers=4, n_heads=8):
        """
        feat_dim : embedding dimension
        window   : sliding window length (e.g., 32)
        """
        super().__init__()
        self.window = window

        # positional embedding for window
        self.pos_embed = nn.Parameter(torch.zeros(1, window, feat_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # transformer encoder
        layer = nn.TransformerEncoderLayer(
            d_model=feat_dim,
            nhead=n_heads,
            dim_feedforward=4 * feat_dim,
            batch_first=True
        )
        self.tr = nn.TransformerEncoder(layer, num_layers=n_layers)

        # value head
        self.value_head = nn.Linear(feat_dim, 1)

    def _get_window(self, seq):
        """
        seq : (B, S, D)
        returns last `window` frames padded to full length
        """
        B, S, D = seq.shape
        W = self.window

        if S >= W:
            return seq[:, S-W:S, :]     # take last W
        else:
            pad = W - S
            # pad by repeating the first frame (or zeros)
            pad_block = seq[:, :1, :].repeat(1, pad, 1)
            return torch.cat([pad_block, seq], dim=1)

    def _transformer_chunked(self, x, chunk_size=128):
        """
        x: (N, W, D)
        returns: (N, W, D)
        """
        outputs = []
        N = x.size(0)

        for i in range(0, N, chunk_size):
            chunk = x[i:i+chunk_size]   # (chunk, W, D)
            out = self.tr(chunk)        # transformer on mini-batch
            outputs.append(out)

        return torch.cat(outputs, dim=0)
    
    def forward(self, X, mask):
        """
        X: (B, S, D)
        mask: ignored (kept for API compatibility)
        returns: (B*S, 1)
        """
        B, S, D = X.shape

        # ---- 1. Build sliding windows ----
        # For each t, we build a window of length W ending at t
        W = self.window

        # Create tensor to hold all windows
        # windows[t] = window ending at timestep t
        windows = []
        for t in range(S):
            seq_t = X[:, :t+1, :]      # (B, t+1, D)
            win_t = self._get_window(seq_t)  # (B, W, D)
            windows.append(win_t)

        # Stack → (B, S, W, D)
        windows = torch.stack(windows, dim=1)

        # reshape into (B*S, W, D)
        flat = windows.reshape(B*S, W, D)

        # ---- 2. Add positional encoding ----
        flat = flat + self.pos_embed

        # ---- 3. Transformer ----
        z = self._transformer_chunked(flat)           # (B*S, W, D)

        # ---- 4. Use last token only ----
        last_token = z[:, -1]       # (B*S, D)

        # ---- 5. Value head ----
        values = self.value_head(last_token) # (B*S, 1)

        return values

class SharedSlidingWindowTransformer(nn.Module):
    def __init__(self, feat_dim: int, window=32, n_layers=4, n_heads=8):
        """
        feat_dim : embedding dimension
        window   : sliding window length (e.g., 32)
        """
        super().__init__()
        self.window = window

        # positional embedding for window
        self.pos_embed = nn.Parameter(torch.zeros(1, window, feat_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # transformer encoder
        layer = nn.TransformerEncoderLayer(
            d_model=feat_dim,
            nhead=n_heads,
            dim_feedforward=4 * feat_dim,
            batch_first=True
        )
        self.tr = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.feat_dim = feat_dim

    def _get_window(self, seq):
        """
        seq : (B, S, D)
        returns last `window` frames padded to full length
        """
        B, S, D = seq.shape
        W = self.window

        if S >= W:
            return seq[:, S-W:S, :]     # take last W
        else:
            pad = W - S
            # pad by repeating the first frame (or zeros)
            pad_block = seq[:, :1, :].repeat(1, pad, 1)
            return torch.cat([pad_block, seq], dim=1)

    def _transformer_chunked(self, x, chunk_size=128):
        """
        x: (N, W, D)
        returns: (N, W, D)
        """
        outputs = []
        N = x.size(0)

        for i in range(0, N, chunk_size):
            chunk = x[i:i+chunk_size]   # (chunk, W, D)
            out = self.tr(chunk)        # transformer on mini-batch
            outputs.append(out)

        return torch.cat(outputs, dim=0)
    
    def forward(self, X, mask):
        """
        X: (B, S, D)
        mask: ignored (kept for API compatibility)
        returns: (B*S, 1)
        """
        B, S, D = X.shape

        # ---- 1. Build sliding windows ----
        # For each t, we build a window of length W ending at t
        W = self.window

        # Create tensor to hold all windows
        # windows[t] = window ending at timestep t
        windows = []
        for t in range(S):
            seq_t = X[:, :t+1, :]      # (B, t+1, D)
            win_t = self._get_window(seq_t)  # (B, W, D)
            windows.append(win_t)

        # Stack → (B, S, W, D)
        windows = torch.stack(windows, dim=1)

        # reshape into (B*S, W, D)
        flat = windows.reshape(B*S, W, D)

        # ---- 2. Add positional encoding ----
        flat = flat + self.pos_embed

        # ---- 3. Transformer ----
        z = self._transformer_chunked(flat)           # (B*S, W, D)

        # ---- 4. Use last token only ----
        last_token = z[:, -1]       # (B*S, D)

        return last_token

class SharedSlidingWindowTransformerActor(nn.Module):
    def __init__(self, transformer: SharedSlidingWindowTransformer, num_action: int):
        super().__init__()
        self.shared_transformer = transformer
        self.policy_head = nn.Linear(transformer.feat_dim, num_action)
        self.window = transformer.window

    def forward(self, X, action_seq, mask):
        X = self.shared_transformer(X, mask)
        return self.policy_head(X)


class SharedSlidingWindowTransformerCritic(nn.Module):
    def __init__(self, transformer: SharedSlidingWindowTransformer):
        super().__init__()
        self.shared_transformer = transformer
        self.value_head = nn.Linear(transformer.feat_dim, 1)
        self.window = transformer.window

    def forward(self, X, mask):
        X = self.shared_transformer(X, mask)
        return self.value_head(X)
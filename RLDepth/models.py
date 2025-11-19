import torch
from torch import nn
import torchvision.models as tv_models

from cons import FEAT_DIM, DEVICE, NUM_ACTIONS


# ---------------------------------------------------------
#  Frozen RGB ResNet Encoder
# ---------------------------------------------------------
class FrozenResNetEncoder(nn.Module):
    """
    Frozen pretrained ResNet encoder that outputs (B, S, feat_dim)
    but processes frames in chunks to avoid GPU OOM.
    Input:  x  (B, S, 3, H, W)
    Output: (B, S, feat_dim)
    """
    def __init__(self, feat_dim: int = FEAT_DIM, device: str = DEVICE, chunk_size: int = 32):
        super().__init__()

        resnet = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)

        # Remove classification head → (B, 512, 1, 1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
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

        # Precompute normalization tensors (ImageNet stats)
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
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

        feats = torch.cat(feats_list, dim=0)    # (B*S, feat_dim)
        return feats.view(b, s, self.feat_dim)  # (B, S, feat_dim)


# ---------------------------------------------------------
#  Depth Encoder (trainable)
# ---------------------------------------------------------
class FrozenResNetDepthEncoder(nn.Module):
    """
    Encodes depth maps (1 x H x W) into feat_dim embeddings.
    Uses a ResNet-18 backbone adapted to 1-channel input.
    We DO NOT freeze this; it is meant to be trained.
    Input:  x  (B, S, 1, H, W)
    Output: (B, S, feat_dim)
    """
    def __init__(self, feat_dim: int = FEAT_DIM, device: str = DEVICE, chunk_size: int = 32):
        super().__init__()

        resnet = tv_models.resnet18(weights=None)

        # Modify first conv: 1 input channel for depth
        resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # (B,512,1,1)
        self.proj = nn.Linear(512, feat_dim)

        self.chunk_size = chunk_size
        self.device = device
        self.feat_dim = feat_dim
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, S, 1, H, W)
        returns: (B, S, feat_dim)
        """
        B, S, _, H, W = x.shape
        x = x.reshape(B * S, 1, H, W)

        outputs = []
        for i in range(0, B * S, self.chunk_size):
            chunk = x[i:i + self.chunk_size].to(self.device)

            f = self.backbone(chunk)   # (chunk, 512, 1, 1)
            f = f.flatten(1)           # (chunk, 512)
            f = self.proj(f)           # (chunk, feat_dim)

            outputs.append(f)

        outputs = torch.cat(outputs, dim=0)
        return outputs.view(B, S, self.feat_dim)


# ---------------------------------------------------------
#  (Optional) LSTM Actor / Critic (unchanged)
# ---------------------------------------------------------
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
        Returns:
            logits: (B*S, num_actions)
        """
        B, S, D = X.shape
        T = self.truncation_len

        h = torch.zeros(self.num_layers, B, D, device=X.device)
        c = torch.zeros(self.num_layers, B, D, device=X.device)
        outputs = []

        for start in range(0, S, T):
            end = min(S, start + T)
            chunk = X[:, start:end, :]   # (B, T, D)

            out, (h, c) = self.lstm(chunk, (h, c))

            h = h.detach()
            c = c.detach()

            outputs.append(out)

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
        Returns:
            values: (B*S, 1)
        """
        B, S, D = X.shape
        T = self.truncation_len

        h = torch.zeros(self.num_layers, B, D, device=X.device)
        c = torch.zeros(self.num_layers, B, D, device=X.device)
        outputs = []

        for start in range(0, S, T):
            end = min(S, start + T)
            chunk = X[:, start:end, :]   # (B, T, D)

            out, (h, c) = self.lstm(chunk, (h, c))

            h = h.detach()
            c = c.detach()

            outputs.append(out)

        outputs = torch.cat(outputs, dim=1)   # (B, S, D)
        logits = self.head(outputs.reshape(B * S, D))
        return logits


# ---------------------------------------------------------
#  Sliding-Window Transformer Actor / Critic
#  (with action embeddings in the Actor)
# ---------------------------------------------------------
class SlidingWindowTransformerActor(nn.Module):
    def __init__(self, feat_dim: int = FEAT_DIM, num_actions: int = NUM_ACTIONS,
                 window=32, n_layers=4, n_heads=8):
        """
        feat_dim : token embedding dimension
        window   : sliding window length
        """
        super().__init__()
        self.window = window
        self.feat_dim = feat_dim
        self.num_actions = num_actions

        # action embedding (same dim as features)
        self.action_embed = nn.Embedding(num_actions, feat_dim)

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

        # policy head (outputs logits over actions)
        self.policy_head = nn.Linear(feat_dim, num_actions)

    def _get_window(self, seq):
        """
        seq : (B, S, D)
        returns last `window` frames padded to full length
        """
        B, S, D = seq.shape
        W = self.window

        if S >= W:
            return seq[:, S - W:S, :]     # take last W
        else:
            pad = W - S
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
            chunk = x[i:i + chunk_size]   # (chunk, W, D)
            out = self.tr(chunk)          # transformer on mini-batch
            outputs.append(out)

        return torch.cat(outputs, dim=0)

    def forward(self, X, actions_seq, mask=None):
        """
        X:           (B, S, D) fused RGB+Depth features
        actions_seq: (B, S)   previous actions (as indices)
        mask:       kept for API compatibility, unused

        returns:
            logits: (B*S, num_actions)
        """
        B, S, D = X.shape
        W = self.window

        # --- 0. Add action embeddings ---
        if actions_seq is not None:
            a_emb = self.action_embed(actions_seq)  # (B, S, D)
            X = X + a_emb

        # --- 1. Build sliding windows ---
        windows = []
        for t in range(S):
            seq_t = X[:, :t + 1, :]          # (B, t+1, D)
            win_t = self._get_window(seq_t)  # (B, W, D)
            windows.append(win_t)

        windows = torch.stack(windows, dim=1)    # (B, S, W, D)

        # reshape into (B*S, W, D)
        flat = windows.reshape(B * S, W, D)

        # --- 2. Add positional encoding ---
        flat = flat + self.pos_embed

        # --- 3. Transformer ---
        z = self._transformer_chunked(flat)           # (B*S, W, D)

        # --- 4. Use last token only ---
        last_token = z[:, -1, :]       # (B*S, D)

        # --- 5. Policy head ---
        logits = self.policy_head(last_token)  # (B*S, num_actions)

        return logits


class SlidingWindowTransformerCritic(nn.Module):
    def __init__(self, feat_dim: int = FEAT_DIM, window=32, n_layers=4, n_heads=8):
        """
        feat_dim : token embedding dimension
        window   : sliding window length
        """
        super().__init__()
        self.window = window
        self.feat_dim = feat_dim

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
            return seq[:, S - W:S, :]
        else:
            pad = W - S
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
            chunk = x[i:i + chunk_size]
            out = self.tr(chunk)
            outputs.append(out)

        return torch.cat(outputs, dim=0)

    def forward(self, X, mask=None):
        """
        X: (B, S, D)
        returns:
            values: (B*S, 1)
        """
        B, S, D = X.shape
        W = self.window

        # 1. sliding windows
        windows = []
        for t in range(S):
            seq_t = X[:, :t + 1, :]
            win_t = self._get_window(seq_t)
            windows.append(win_t)

        windows = torch.stack(windows, dim=1)  # (B, S, W, D)
        flat = windows.reshape(B * S, W, D)

        # 2. add positional encoding
        flat = flat + self.pos_embed

        # 3. transformer
        z = self._transformer_chunked(flat)  # (B*S, W, D)

        # 4. last token
        last_token = z[:, -1, :]  # (B*S, D)

        # 5. value head
        values = self.value_head(last_token)  # (B*S, 1)

        return values

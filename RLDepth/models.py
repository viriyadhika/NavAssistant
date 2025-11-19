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

    def forward(self, x):
            """
            x : (B, S, 1, H, W)
            """
            B, S, _, H, W = x.shape
            x = x.reshape(B * S, 1, H, W)

            feats_list = []
            N = B * S

            for i in range(0, N, self.chunk_size):
                chunk = x[i:i + self.chunk_size].to(self.device)

                f = self.cnn(chunk)         # (chunk, 256, 1, 1)
                f = f.flatten(1)            # (chunk, 256)
                f = self.proj(f)            # (chunk, feat_dim)

                feats_list.append(f)

            feats = torch.cat(feats_list, dim=0)
            return feats.view(B, S, self.feat_dim)

# ---------------------------------------------------------
#  Depth Encoder (trainable)
# ---------------------------------------------------------
class ConvDepthEncoder(nn.Module):
    """
    Encodes depth maps (1 x H x W) into feat_dim embeddings using a lightweight ConvNet.
    No chunking — processes B*S frames directly.

    Input:  x  (B, S, 1, H, W)
    Output: (B, S, feat_dim)
    """
    def __init__(self, feat_dim: int = FEAT_DIM, device: str = DEVICE):
        super().__init__()

        # Lightweight ConvNet for depth
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),   # -> (32, H/2, W/2)
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> (64, H/4, W/4)
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # -> (128, H/8, W/8)
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),# -> (256, H/16, W/16)
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),                          # -> (256,1,1)
        )

        # Final projection to the shared embedding dimension
        self.proj = nn.Linear(256, feat_dim)

        self.feat_dim = feat_dim
        self.device = device
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, S, 1, H, W)
        returns: (B, S, feat_dim)
        """
        B, S, _, H, W = x.shape

        # Flatten time dimension into batch
        x = x.view(B * S, 1, H, W)  # (B*S, 1, H, W)

        f = self.cnn(x)             # (B*S, 256, 1, 1)
        f = f.flatten(1)            # (B*S, 256)
        f = self.proj(f)            # (B*S, feat_dim)

        return f.view(B, S, self.feat_dim)


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
        self.fuse = nn.Linear(2*feat_dim, feat_dim)

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
        D = D // 2 # Because X is fused between RGB and D channel
        W = self.window

        # --- 0. Add action embeddings ---
        if actions_seq is not None:
            a_emb = self.action_embed(actions_seq)  # (B, S, D)
            X = self.fuse(X) + a_emb

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
        self.fuse = nn.Linear(2*feat_dim, feat_dim)
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

        # Normalize back to FEATURE_DIM
        X = self.fuse(X)
        D = D // 2

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

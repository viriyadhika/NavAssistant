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
    
class FrozenCLIPEncoder(nn.Module):
    """
    Frozen CLIP image encoder: runs the visual transformer in FP16
    for speed, then converts output embeddings back to float32 for stability.
    Returns (B, S, feat_dim), same as your CNN encoder.
    """
    def __init__(self, feat_dim: int = 512, device: str = DEVICE, clip_model_name: str = "ViT-B/32"):
        super().__init__()
        model, _ = clip.load(clip_model_name, device=device)
        self.clip_model = model.visual.eval().float()
        self.device = device

        for p in self.clip_model.parameters():
            p.requires_grad = False

        # Detect output dimension
        if hasattr(self.clip_model, "proj"):
            clip_out_dim = self.clip_model.proj.shape[1]
        else:
            clip_out_dim = getattr(self.clip_model, "output_dim", 512)

        # Projection to PPO's feature dim
        self.proj = nn.Linear(clip_out_dim, feat_dim).to(device)  # ensure FP32 projection
        self.feat_dim = feat_dim

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, S, 3, H, W), values in [0,1]
        returns: (B, S, feat_dim) as float32
        """
        b, s, c, h, w = x.shape
        x = x.reshape(b * s, c, h, w).to(self.device, dtype=torch.float32)

        # Resize to CLIP resolution
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        # Normalize (CLIP standard mean/std)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        # Forward through CLIP (FP16)
        feats = self.clip_model(x)  # (B*S, clip_out_dim), in half precision

        # Project to feature dim
        z = self.proj(feats)  # (B*S, feat_dim) float32
        z = z.view(b, s, self.feat_dim)
        return z

class Actor(nn.Module):
    def __init__(self, feat_dim, num_actions, n_layers=2, n_heads=8, max_len=EPISODE_STEPS):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_actions = num_actions

        # Positional embedding for temporal order
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, feat_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Action embedding (one vector per discrete action)
        self.action_embed = nn.Embedding(num_actions + 1, feat_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim, nhead=n_heads, dim_feedforward=4 * feat_dim, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Policy head
        self.policy_head = nn.Linear(feat_dim, num_actions)

    def forward(self, feats_seq, actions_seq, mask=None):
        """
        feats_seq: (B, S, feat_dim)
        actions_seq: (B, S)  -- discrete action indices (use 0 for first timestep)
        mask: optional transformer mask
        """
        b, s, _ = feats_seq.shape
        feats_seq = feats_seq + self.pos_embed[:, :s, :]

        # Embed actions and fuse with features
        action_emb = self.action_embed(actions_seq)  # (B, S, feat_dim)
        fused = feats_seq + action_emb               # elementwise fusion

        # Transformer
        z = self.transformer(fused, mask).view(b * s, -1)

        # Policy logits
        logits = self.policy_head(z)
        return logits


class Critic(nn.Module):
    def __init__(self, feat_dim, n_layers=2, n_heads=8, max_len=EPISODE_STEPS):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, feat_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim, nhead=n_heads, dim_feedforward=4*feat_dim, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.policy_head = nn.Linear(feat_dim, 1)

    def forward(self, feats_seq, mask):
        # feats_seq: (B, seq_len, feat_dim)
        b, s, _ = feats_seq.shape
        feats_seq = feats_seq + self.pos_embed[:,:s,:]
        z = self.transformer(feats_seq, mask).view(b*s, -1)  # (B*seq_len, feat_dim)
        logits = self.policy_head(z)
        return logits


class LSTMActor(nn.Module):
    def __init__(self, feat_dim: int, num_actions: int):
        super().__init__()
        self.num_layers = 8
        self.hidden_size = feat_dim
        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=feat_dim,
            num_layers=self.num_layers,
            batch_first=True
        )
        self.head = nn.Linear(feat_dim, num_actions)

    def forward(self, X, mask=None):
        B, S, D = X.shape

        # initialize hidden states PER BATCH
        h0 = torch.zeros(self.num_layers, B, D, device=X.device)
        c0 = torch.zeros(self.num_layers, B, D, device=X.device)

        out, _ = self.lstm(X, (h0, c0))   # out: (B, S, D)
        out = out.reshape(B * S, D)
        return self.head(out)


class LSTMCritic(nn.Module):
    def __init__(self, feat_dim: int):
        super().__init__()
        self.num_layers = 8
        self.hidden_size = feat_dim
        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=feat_dim,
            num_layers=self.num_layers,
            batch_first=True
        )
        self.head = nn.Linear(feat_dim, 1)

    def forward(self, X, mask=None):
        B, S, D = X.shape

        # initialize hidden states PER BATCH
        h0 = torch.zeros(self.num_layers, B, D, device=X.device)
        c0 = torch.zeros(self.num_layers, B, D, device=X.device)

        out, _ = self.lstm(X, (h0, c0))   # out: (B, S, D)
        out = out.reshape(B * S, D)
        return self.head(out)


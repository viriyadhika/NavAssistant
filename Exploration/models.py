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
    Frozen pretrained ResNet encoder that outputs (B, S, feat_dim),
    identical to CNNEncoder's output shape.
    """
    def __init__(self, feat_dim: int = FEAT_DIM, device: str = DEVICE):
        super().__init__()
        # Use a small ResNet backbone
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Remove classification head (keep up to global avgpool)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # -> (B, 512, 1, 1)
        self.backbone.eval()  # important: freeze batchnorm behavior

        # Freeze all backbone parameters
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Save output dimension of resnet conv feature
        self.backbone_out_dim = resnet.fc.in_features  # typically 512

        # Small projection layer to match CNN's feat_dim
        self.proj = nn.Linear(self.backbone_out_dim, feat_dim)
        self.feat_dim = feat_dim

        self.device = device
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, S, C, H, W)
        returns: (B, S, feat_dim)
        """
        b, s, c, h, w = x.shape
        x = x.reshape(b * s, c, h, w).to(self.device)

        # Normalize to match ImageNet statistics
        # Expected input range: [0,1]
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1,3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1,3,1,1)
        x = (x - mean) / std

        with torch.no_grad():  # frozen backbone
            feats = self.backbone(x)        # (B*S, 512, 1, 1)
            feats = feats.flatten(1)        # (B*S, 512)

        z = self.proj(feats)                # (B*S, feat_dim)
        z = z.view(b, s, self.feat_dim)     # reshape to (B, S, D)
        return z
    
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
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, feat_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim, nhead=n_heads, dim_feedforward=4*feat_dim, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.policy_head = nn.Linear(feat_dim, num_actions)

    def forward(self, feats_seq, mask):
        # feats_seq: (B, seq_len, feat_dim)
        b, s, _ = feats_seq.shape
        feats_seq = feats_seq + self.pos_embed[:,:s,:]
        z = self.transformer(feats_seq, mask).view(b*s, -1)  # (B*seq_len, feat_dim)
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

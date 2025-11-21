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
from cons import DEVICE, GAMMA, GAE_LAMBDA, LR, PPO_CLIP, TRAIN_EPOCHS, VALUE_COEF, ENTROPY_COEF, MAX_GRAD_NORM, EPISODE_STEPS, INTRINSIC_COEF, EXTRINSIC_COEF, ACTIONS, transform
from models import ActorCritic
import clip
import random
from torchvision import transforms as TF

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

def preprocess_depth(depth_np):
    # depth_np: (H, W) float numpy
    depth = torch.from_numpy(depth_np.copy()).float()        # (H,W)
    depth = depth.unsqueeze(0).unsqueeze(0)                  # (1,1,H,W)
    depth = F.interpolate(depth, size=(224, 224), mode="nearest")
    depth = depth / 10.0                                     # or your scale
    return depth


class ExtrinsicReward:
    """
    Keeps last N agent positions and gives reward 
    based on distance from the sliding-window mean position.

    reward = alpha * || pos_t - mean(last_positions) ||

    If fewer than N positions exist, uses existing ones.

    You can also add optional penalties or bonuses for movement success.
    """

    def __init__(
        self,
        window_size=32,
        alpha=3.0,
        fail_penalty=-0.1,
        move_bonus=0.05
    ):
        self.window_size = window_size
        self.alpha = alpha

        # optional bonuses/penalties
        self.fail_penalty = fail_penalty
        self.move_bonus = move_bonus

        # store the last N 2D positions
        self.positions = deque(maxlen=window_size)

    def reset(self):
        """Clear window at start of an episode."""
        self.positions.clear()

    def compute(self, event):
        """
        event: AI2-THOR event
        Returns extrinsic reward based on:
          (1) novelty movement reward
          (2) small movement bonus
          (3) failure penalty
        """

        # ------------------------
        #  Extract agent position
        # ------------------------
        pos_data = event.metadata["agent"]["position"]
        pos = np.array([pos_data["x"], pos_data["z"]], dtype=np.float32)

        # ------------------------
        #  Compute sliding window mean
        # ------------------------
        if len(self.positions) == 0:
            self.positions.append(pos)
            movement_novelty = 0.0
        else:
            mean_pos = np.mean(self.positions, axis=0)
            movement_novelty = float(np.linalg.norm(pos - mean_pos))
            self.positions.append(pos)

        # weight the novelty term
        movement_reward = self.alpha * movement_novelty

        # ------------------------
        #  Optional action-based terms
        # ------------------------
        fail = not event.metadata.get("lastActionSuccess", True)

        # small failure penalty
        penalty = self.fail_penalty if fail else 0.0

        return movement_reward + penalty


class CLIPCuriosity:
    """
    CLIP-based curiosity:
      - Encodes frames with CLIP
      - Computes novelty = 1 - mean(top-k cosine sim)
      - Normalizes novelty with running mean/std
    """
    def __init__(
        self,
        device=DEVICE,
        model_name="ViT-B/32",
        buffer_size: int = 10_000,
        topk: int = 5,
        ema_beta: float = 0.99,
        reward_scale: float = 1.0,
        every_n_steps: int = 1,
    ):
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()

        self.buffer = deque(maxlen=buffer_size)
        self.topk = topk
        self.ema_beta = ema_beta
        self.reward_scale = reward_scale
        self.every_n_steps = every_n_steps

        self.novelty_mean = 0.0
        self.novelty_var = 1.0
        self.step_count = 0

    @torch.no_grad()
    def encode_frame(self, frame_np: np.ndarray) -> torch.Tensor:
        """
        frame_np: (H,W,3) uint8
        returns:  (D,) normalized CLIP embedding
        """
        img = Image.fromarray(frame_np)
        img_t = self.preprocess(img).unsqueeze(0).to(self.device)  # (1,3,224,224)
        emb = self.model.encode_image(img_t)                       # (1,D)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.squeeze(0)                                      # (D,)

    @torch.no_grad()
    def compute_intrinsic_reward(self, frame_np: np.ndarray) -> float:
        self.step_count += 1

        # Optionally skip some steps to save compute
        if self.step_count % self.every_n_steps != 0:
            return 0.0

        emb = self.encode_frame(frame_np)  # (D,)

        if len(self.buffer) == 0:
            self.buffer.append(emb)
            return 0.0

        past = torch.stack(list(self.buffer), dim=0)  # (N,D)
        sims = (emb @ past.T)                         # (N,)

        k = min(self.topk, sims.size(0))
        topk_sim = torch.topk(sims, k=k, largest=True).values.mean().item()
        novelty = max(0.0, 1.0 - topk_sim)           # [0,1]

        # Update running mean / var (Welford-style EMA)
        # mean_t = beta*mean_{t-1} + (1-beta)*x_t
        # var_t  = beta*var_{t-1} + (1-beta)*(x_t - mean_t)^2
        old_mean = self.novelty_mean
        self.novelty_mean = self.ema_beta * self.novelty_mean + (1 - self.ema_beta) * novelty
        self.novelty_var  = self.ema_beta * self.novelty_var  + (1 - self.ema_beta) * (novelty - self.novelty_mean) ** 2

        std = float(self.novelty_var ** 0.5 + 1e-8)
        normalized_novelty = (novelty - self.novelty_mean) / std

        self.buffer.append(emb.detach())
        return float(normalized_novelty * self.reward_scale)

    def reset(self):
        self.buffer.clear()
        self.novelty_mean = 0.0
        self.novelty_var = 1.0
        self.step_count = 0



class VGGTCuriosity:
    """
    VGGT-based curiosity:
      - Encodes frames using VGGT aggregator tokens
      - Computes novelty = 1 - mean(top-k cosine similarity)
      - Normalizes novelty with running mean/std (EMA)
      - Drop-in replacement for CLIPCuriosity
    """

    def __init__(
        self,
        device="cuda",
        buffer_size: int = 10_000,
        topk: int = 5,
        ema_beta: float = 0.99,
        reward_scale: float = 1.0,
        every_n_steps: int = 1,
    ):
        self.device = device

        # Load VGGT (1B parameters)
        self.model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
        self.model.eval()

        # rolling embedding memory
        self.buffer = deque(maxlen=buffer_size)

        self.topk = topk
        self.ema_beta = ema_beta
        self.reward_scale = reward_scale
        self.every_n_steps = every_n_steps

        # EMA stats
        self.novelty_mean = 0.0
        self.novelty_var = 1.0
        self.step_count = 0

    def preprocess_vggt_from_array(self, frame_np, mode="crop", target_size=224):
        """
        VGGT-compatible preprocessing directly from numpy array (H,W,3).
        Returns tensor of shape (1,3,H,W).
        """
        # Convert to PIL
        img = Image.fromarray(frame_np)

        # Ensure RGB
        if img.mode != "RGB":
            img = img.convert("RGB")

        width, height = img.size

        if mode == "pad":
            # Largest dimension → target_size
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14

        else:  # crop (same as official code)
            new_width = target_size
            new_height = round(height * (new_width / width) / 14) * 14

        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)

        # To tensor [0,1]
        img_t = TF.ToTensor()(img)

        # center crop height if too tall (crop mode only)
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img_t = img_t[:, start_y:start_y + target_size, :]

        # pad if needed (pad mode)
        if mode == "pad":
            h_pad = target_size - img_t.shape[1]
            w_pad = target_size - img_t.shape[2]
            if h_pad > 0 or w_pad > 0:
                pad_top = h_pad // 2
                pad_bottom = h_pad - pad_top
                pad_left = w_pad // 2
                pad_right = w_pad - pad_left
                img_t = F.pad(img_t,
                            (pad_left, pad_right, pad_top, pad_bottom),
                            value=1.0)  # white padding

        return img_t.unsqueeze(0)   # (1,3,H,W)

    # ----------------------------------------------------
    #  Encode single frame with VGGT aggregator
    # ----------------------------------------------------
    @torch.no_grad()
    def encode_frame(self, frame_np: np.ndarray) -> torch.Tensor:
        """
        frame_np: np array (H, W, 3), uint8
        returns: 1 vector (D,)
        """
        # Convert array to PIL

        # Preprocess into (N, 3, H, W)
        imgs = self.preprocess_vggt_from_array(frame_np).to(self.device)  # (1,3,H,W)

        # Add sequence dimension → (1,1,3,H,W)
        imgs = imgs.unsqueeze(0).to(self.device)

        # Run VGGT aggregator
        tokens_list, _ = self.model.aggregator(imgs)
        tokens = tokens_list[-1]  # (1,1,N,D)

        # Mean-pool across tokens
        emb = tokens.mean(dim=2).squeeze(0).squeeze(0)  # (D,)

        # Normalize the embedding
        emb = emb / emb.norm(dim=-1, keepdim=True)

        return emb
    # ----------------------------------------------------
    #  Curiosity reward from VGGT embeddings
    # ----------------------------------------------------
    @torch.no_grad()
    def compute_intrinsic_reward(self, frame_np: np.ndarray) -> float:
        self.step_count += 1

        # optional step skipping
        if self.step_count % self.every_n_steps != 0:
            return 0.0

        # Embed current frame
        emb = self.encode_frame(frame_np)  # (D,)

        if len(self.buffer) == 0:
            self.buffer.append(emb)
            return 0.0

        # Stack past embeddings
        past = torch.stack(list(self.buffer), dim=0)  # (N, D)

        # cosine similarity to all past embeddings
        sims = (emb @ past.T)                         # (N,)

        # top-k similarity
        k = min(self.topk, sims.size(0))
        topk_sim = torch.topk(sims, k=k, largest=True).values.mean().item()

        novelty = max(0.0, 1.0 - topk_sim)  # in [0,1]

        # ----------------------------------------
        # EMA normalization (Welford-style)
        # ----------------------------------------
        old_mean = self.novelty_mean

        self.novelty_mean = self.ema_beta * self.novelty_mean + (1 - self.ema_beta) * novelty
        self.novelty_var  = self.ema_beta * self.novelty_var  + (1 - self.ema_beta) * (novelty - self.novelty_mean) ** 2

        std = float(self.novelty_var ** 0.5 + 1e-8)

        normalized = (novelty - self.novelty_mean) / std

        # Update memory
        self.buffer.append(emb.detach())

        return float(normalized * self.reward_scale)

    # ----------------------------------------------------
    def reset(self):
        self.buffer.clear()
        self.novelty_mean = 0.0
        self.novelty_var = 1.0
        self.step_count = 0


class RNDCuriosity:
    """
    Drop-in replacement for CLIPCuriosity / VGGTCuriosity.
    Uses Random Network Distillation (RND):
      intrinsic_reward = prediction_error(target(x), predictor(x))
    """

    def __init__(
        self,
        device="cuda",
        emb_dim=256,
        lr=1e-4,
        ema_beta=0.99,
        reward_scale=1.0,
        every_n_steps=1,
    ):
        self.device = device
        self.emb_dim = emb_dim
        self.lr = lr
        self.ema_beta = ema_beta
        self.reward_scale = reward_scale
        self.every_n_steps = every_n_steps
        self.step_count = 0

        # ===== Build conv encoder (same for target & predictor) =====
        self.target = self._build_encoder().to(device)
        self.predictor = self._build_encoder().to(device)

        # Freeze target
        for p in self.target.parameters():
            p.requires_grad = False

        # Optimizer for predictor
        self.opt = torch.optim.Adam(self.predictor.parameters(), lr=lr)

        # Running stats for reward normalization
        self.r_mean = 0.0
        self.r_var = 1.0

        # Compute correct flatten dimension dynamically
        self.flat_dim = self._get_flat_dim()

        # Replace final linear layers with correct sizes
        self.target.fc = nn.Linear(self.flat_dim, emb_dim).to(device)
        self.predictor.fc = nn.Linear(self.flat_dim, emb_dim).to(device)

    # ----------------------------------------------------
    def _build_encoder(self):
        """Conv encoder producing some spatial feature map, then flatten + fc."""
        return nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Identity()   # placeholder for fc replacement
        )

    # ----------------------------------------------------
    def _get_flat_dim(self):
        """Pass a dummy tensor to determine conv flatten size."""
        dummy = torch.zeros(1, 3, 224, 224).to(self.device)
        x = self.target[:-1](dummy)    # run without the final fc
        return x.numel()

    # ----------------------------------------------------
    @torch.no_grad()
    def encode_frame(self, frame_np):
        """Return  embedding from the TARGET network (not predictor)."""
        x = transform(frame_np).unsqueeze(0).to(self.device)  # (1,3,224,224)

        with torch.no_grad():
            h = self.target[:-1](x)        # conv → flatten
            h = h.view(1, -1)
            emb = self.target.fc(h)        # (1,emb_dim)
            emb = F.normalize(emb, dim=-1)

        return emb.squeeze(0)              # (emb_dim,)

    # ----------------------------------------------------
    def compute_intrinsic_reward(self, frame_np):
        self.step_count += 1

        if self.step_count % self.every_n_steps != 0:
            return 0.0

        x = transform(frame_np).unsqueeze(0).to(self.device)  # (1,3,224,224)

        # -------- Target --------------
        with torch.no_grad():
            t = self.target[:-1](x).view(1, -1)
            t = self.target.fc(t)

        # -------- Predictor -----------
        p = self.predictor[:-1](x).view(1, -1)
        p = self.predictor.fc(p)

        # -------- Prediction error ----
        error = F.mse_loss(p, t.detach(), reduction="none").mean()
        error_val = error.item()

        # -------- Train predictor -----
        self.opt.zero_grad()
        error.backward()
        self.opt.step()

        # -------- Normalize reward ----
        self.r_mean = self.ema_beta * self.r_mean + (1 - self.ema_beta) * error_val
        self.r_var  = self.ema_beta * self.r_var  + (1 - self.ema_beta) * (error_val - self.r_mean)**2

        std = max(self.r_var**0.5, 1e-6)
        norm_reward = (error_val - self.r_mean) / std

        return float(norm_reward * self.reward_scale)

    # ----------------------------------------------------
    def reset(self):
        self.step_count = 0
        self.r_mean = 0.0
        self.r_var = 1.0

class RolloutBuffer:
    def __init__(self):
        self.feats = []   # list of (feat_dim,) tensors
        self.actions = []
        self.logps = []
        self.rewards = []
        self.values = []
        self.dones = []

    def add(self, feat, action, logp, reward, value, done):
        # feat: (feat_dim,) on CPU or GPU
        self.feats.append(feat.detach().cpu())
        self.actions.append(int(action))
        self.logps.append(float(logp))
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.dones.append(bool(done))

    def __len__(self):
        return len(self.rewards)

    def to_tensors(self, device=DEVICE):
        feats = torch.stack(self.feats, dim=0).to(device)          # (T,D)
        actions = torch.tensor(self.actions, dtype=torch.long, device=device)  # (T,)
        logps = torch.tensor(self.logps, dtype=torch.float32, device=device)   # (T,)
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=device)  # (T,)
        values = torch.tensor(self.values, dtype=torch.float32, device=device)    # (T,)
        dones = torch.tensor(self.dones, dtype=torch.float32, device=device)      # (T,)
        return feats, actions, logps, rewards, values, dones


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = GAMMA,
    lam: float = GAE_LAMBDA,
):
    """
    rewards, values, dones: (T,)
    returns:
        advantages: (T,)
        returns:    (T,)
    """
    T = rewards.size(0)
    advantages = torch.zeros(T, dtype=torch.float32, device=rewards.device)
    last_adv = 0.0

    # Append bootstrap value V_{T} = 0 (or pass last value externally)
    values_ext = torch.cat([values, torch.zeros(1, device=values.device)], dim=0)

    for t in reversed(range(T)):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * values_ext[t + 1] * nonterminal - values_ext[t]
        last_adv = delta + gamma * lam * nonterminal * last_adv
        advantages[t] = last_adv

    returns = advantages + values
    return advantages, returns


class ThorNavEnv:
    """
    Thin wrapper around ai2thor.Controller to:
      - map discrete action indices to ACTION strings
      - compute intrinsic CLIP reward
      - optionally extrinsic reward from custom function
    """
    def __init__(
        self,
        controller,
        clip_curiosity: CLIPCuriosity,
        extrinsic_reward: ExtrinsicReward,
        pos_bonus_coef: float = 0.01,
    ):
        self.controller = controller
        self.clip_curiosity = clip_curiosity
        self.extrinsic_reward = extrinsic_reward
        self.pos_bonus_coef = pos_bonus_coef

        self.step_count = 0

    def reset(self, init_position: dict = None):
        """
        If init_position is None, do a random TeleportFull to a reachable spot.
        """
        self.clip_curiosity.reset()
        self.extrinsic_reward.reset()
        self.step_count = 0

        event = self.controller.step("GetReachablePositions")
        reachable = event.metadata["actionReturn"]

        if init_position is None:
            target = random.choice(reachable)
        else:
            target = init_position

        event = self.controller.step(
            action="TeleportFull",
            x=target["x"],
            y=target["y"],
            z=target["z"],
            rotation={"x": 0, "y": 0, "z": 0},
            horizon=0,
            standing=True,
        )
        return event

    def step(self, action_idx: int):
        """
        Returns:
            event, total_reward, done
        """
        self.step_count += 1
        action_str = ACTIONS[action_idx]
        event = self.controller.step(action_str)

        # intrinsic reward from CLIP
        intrinsic_r = self.clip_curiosity.compute_intrinsic_reward(event.frame.copy())

        # simple extrinsic reward (optional user-defined)
        extrinsic_r = float(self.extrinsic_reward.compute(event))

        total_reward = ((
            INTRINSIC_COEF * intrinsic_r +
            EXTRINSIC_COEF * extrinsic_r
        ) - 0.8) / 20

        done = False  # you can define a terminal condition if you want
        return event, total_reward, done
    

class PPOTrainer:
    def __init__(
        self,
        actor_critic: ActorCritic,
        lr: float = LR,
        clip_eps: float = PPO_CLIP,
        value_coef: float = VALUE_COEF,
        entropy_coef: float = ENTROPY_COEF,
        max_grad_norm: float = MAX_GRAD_NORM,
        device: str = DEVICE,
    ):
        self.ac = actor_critic
        self.optimizer = torch.optim.Adam([
            {'params': self.ac.gru.parameters(), 'lr': lr * 0.125},
            {'params': self.ac.policy_head.parameters(), 'lr': lr},
            {'params': self.ac.value_head.parameters(), 'lr': lr},
        ])
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = device

    def collect_rollout(
        self,
        env: ThorNavEnv,
        horizon: int = EPISODE_STEPS,
    ) -> Tuple[RolloutBuffer, float]:
        buf = RolloutBuffer()
        total_episode_reward = 0.0

        # reset env
        event = env.reset()
        h = self.ac.init_hidden(batch_size=1)

        for t in range(horizon):
            # --- Build normalized RGB tensor: (1,3,H,W) ---
            rgb = transform(event.frame.copy()).unsqueeze(0).to(self.device)  # [0,1]

            # depth frame: numpy (H,W) float
            depth_t = preprocess_depth(event.depth_frame.copy()).to(self.device)  # (1,1,H,W)

            # --- Encode and act ---
            feat = self.ac.encode_obs(rgb, depth_t)           # (1,D)
            feat_seq = feat.unsqueeze(1)                      # (1,1,D)
            action, logp, value, h = self.ac.act(feat_seq, h)

            # --- Step env ---
            event, reward, done = env.step(action)
            done = t == horizon - 1

            # --- Store ---
            buf.add(
                feat=feat.squeeze(0).detach().cpu(),  # (D,)
                action=action,
                logp=logp,
                reward=reward,
                value=value,
                done=done,
            )

            total_episode_reward += reward

            if done:
                # For now we just continue collecting until horizon,
                # but you could break and re-reset if you define dones.
                event = env.reset()
                h = self.ac.init_hidden(batch_size=1)

        return buf, total_episode_reward

    def ppo_update(self, buf, epochs=TRAIN_EPOCHS, is_pretrain=False):
        feats, actions, old_logps, rewards, values, dones = buf.to_tensors(self.device)

        feats_seq = feats.unsqueeze(0)
        actions_seq = actions

        advantages, returns = compute_gae(rewards, values, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # If pretraining critic → freeze actor
        self.ac.policy_head.requires_grad_(not is_pretrain)

        # Expand epochs for warmup
        if is_pretrain:
            epochs *= 5

        for ep in range(epochs):
            h0 = self.ac.init_hidden(1)

            logits, value_pred, _ = self.ac.forward_sequence(feats_seq, h0)
            logits = logits.squeeze(0)
            value_pred = value_pred.squeeze(0)

            value_loss = F.mse_loss(value_pred, returns)

            if is_pretrain:
                loss = value_loss
            else:
                dist = Categorical(logits=logits)
                logps = dist.log_prob(actions_seq)
                entropy = dist.entropy()

                ratio = torch.exp(logps - old_logps)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages

                policy_loss = -torch.min(surr1, surr2).mean()
                entropy_bonus = entropy.mean()

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_bonus

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
            self.optimizer.step()

            if ep == epochs - 1:
                with torch.no_grad():
                    approx_kl = (old_logps - logps).mean().item()
                print(
                    f"[PPO] Epoch {ep+1}/{epochs} "
                    f"Loss={loss.item():.4f} "
                    f"Policy={policy_loss.item():.4f} "
                    f"Value={value_loss.item():.4f} "
                    f"Entropy={entropy_bonus.item():.4f} "
                    f"KL={approx_kl:.4f}"
                )

@torch.no_grad()
def run_inference(
    actor_critic,
    env,
    max_steps=500,
    deterministic=False,
    device=DEVICE,
):
    """
    Run a single inference episode.
    
    Args:
        actor_critic : trained ActorCritic model
        env          : ThorNavEnv with curiosity disabled
        max_steps    : max steps per episode
        deterministic: if True -> greedy action (argmax)
                       if False -> sample from policy
    """
    actor_critic.eval()

    # Reset environment and hidden state
    event = env.reset()
    h = actor_critic.init_hidden(batch_size=1)

    total_reward = 0.0
    trajectory = []  # store (frame, action, reward)

    for t in range(max_steps):

        # ---- RGB ----
        rgb = transform(event.frame.copy()).unsqueeze(0).to(device)

        # ---- Depth ----
        depth_t = preprocess_depth(event.depth_frame.copy()).to(device)

        # ---- Encode ----
        feat = actor_critic.encode_obs(rgb, depth_t)       # (1, D)
        feat_seq = feat.unsqueeze(1)                       # (1, 1, D)

        # ---- Get policy ----
        logits, values, h_new = actor_critic.forward_sequence(feat_seq, h)
        logits = logits[:, -1, :]     # (1, num_actions)
        value = values[:, -1]         # (1,)

        if deterministic:
            action = torch.argmax(logits, dim=-1).item()
        else:
            # sample from policy
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample().item()

        # ---- Step env ----
        event, reward, done = env.step(action)

        # ---- Save info ----
        trajectory.append({
            "frame": event.frame.copy(),
            "action": action,
            "reward": reward,
            "value": float(value.item()),
            "logits": logits
        })

        total_reward += reward
        h = h_new

        if done:
            break

    return trajectory, total_reward

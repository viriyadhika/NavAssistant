import torch
from torch import nn
from cons import NUM_ACTIONS, FEAT_DIM, DEVICE, LR, transform, ACTIONS, GAMMA, GAE_LAMBDA, TRAIN_EPOCHS, MINIBATCHES, PPO_CLIP, VALUE_COEF, MAX_GRAD_NORM, EPISODE_STEPS
import os
import clip
import torch.nn.functional as F
from collections import deque
from PIL import Image
import numpy as np
from abc import ABC, abstractmethod

class RolloutBuffer:
    def __init__(self):
        self.obs, self.actions, self.logps = [], [], []
        self.rewards, self.values, self.dones = [], [], []
    def add(self, obs, action_idx, logp, reward, value, done):
        self.obs.append(obs.cpu())              # store CPU to save VRAM
        self.actions.append(int(action_idx))
        self.logps.append(float(logp))
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.dones.append(bool(done))
    def __len__(self): 
        return len(self.rewards)

class ActorCritic():
    def __init__(self, encoder: nn.Module, actor: nn.Module, critic: nn.Module):
        self.actor_critic_encoder = encoder.to(DEVICE)
        self.actor   = actor.to(DEVICE)
        self.critic  = critic.to(DEVICE)
        self.optimizer = torch.optim.AdamW(
            list(self.actor_critic_encoder.parameters()) + list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=LR
        )

# ------------------------ Model (shared encoder) ------------------------
class CNNEncoder(nn.Module):
    """Shape-safe: no hardcoded flatten size; uses GAP -> Linear(feat_dim)."""
    def __init__(self, feat_dim: int, n_hidden: int):
        super().__init__()
        self.feat_dim = feat_dim
        self.conv = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 5, 2, padding=2),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # /4
        
            # Block 2
            nn.Conv2d(32, 64, 5, 1, padding=2),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # /8

            nn.Conv2d(64, 64, 5, 2, padding=2),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # /16

            *self._block(n_hidden)
        )
        
        self.gap  = nn.AdaptiveAvgPool2d((1,1))
        self.proj = nn.Linear(64, feat_dim)

    def _block(self, n_hidden: int):
        return [
            layer
            for _ in range(n_hidden)
            for layer in (
                nn.Conv2d(64, 64, 3, 1, padding=1),
                nn.GroupNorm(8, 64),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
        ]

    def forward(self, x):
        b, s, c, h, w = x.shape
        x = x.reshape(b*s, c, h, w)
        z = self.conv(x)                # (B*S,64,h,w)
        z = self.gap(z).squeeze(-1).squeeze(-1)  # (B*S,S,64)
        z = self.proj(z)                # (B*S,S,feat_dim)
        return z.view(b, s, self.feat_dim)



class RNDIntrinsicReward(nn.Module):
    """
    Random Network Distillation:
    Fixed (random) target; train predictor to match it.
    Intrinsic reward = prediction error (MSE), normalized online.
    """
    def __init__(self, feat_dim=FEAT_DIM, n_hidden=2, lr=1e-5, device=DEVICE, ema_beta=0.99, eps=1e-8):
        super().__init__()
        self.device = device
        self.feat_dim = feat_dim
        self.n_hidden = n_hidden
        self.rnd_mean = 0.0
        self.rnd_var  = 1.0
        self.ema_beta = ema_beta
        self.eps = eps

        # Fixed target (no grad)
        self.target = CNNEncoder(feat_dim, n_hidden).to(device)
        for p in self.target.parameters():
            p.requires_grad = False
        self.target.eval()

        # Trainable predictor
        self.predictor = CNNEncoder(feat_dim, n_hidden).to(device)
        self.predictor.train()

        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=lr)
        self.mse = nn.MSELoss(reduction='none')
        self.lr = lr

    def compute_reward(self, obs_t: torch.Tensor) -> float:
        """
        obs_t: (3,H,W) or (1,3,H,W); CNNEncoder handles unsqueeze.
        Returns scalar normalized intrinsic reward.
        """
        with torch.no_grad():
            tgt = self.target(obs_t)              # (B, feat_dim)

        pred = self.predictor(obs_t)              # (B, feat_dim)
        
        err_vec = (pred - tgt).pow(2).mean(dim=-1)  # (1,)
        raw = err_vec.detach().mean()               # scalar tensor

        # Update running mean/var (EMA)
        m = float(raw)
        self.rnd_mean = self.ema_beta * self.rnd_mean + (1 - self.ema_beta) * m
        diff = m - self.rnd_mean
        self.rnd_var  = self.ema_beta * self.rnd_var + (1 - self.ema_beta) * (diff * diff)
        std = (self.rnd_var ** 0.5) + self.eps

        # Train predictor
        loss = err_vec.mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        # Normalized intrinsic reward
        return float((m - self.rnd_mean) / std)

    
    def pre_process_rnd(self, event):
        obs_t = transform(event.frame.copy()).to(DEVICE)
        return obs_t.unsqueeze(0).unsqueeze(0)
    
    def reset(self):
        self.predictor = CNNEncoder(self.feat_dim, self.n_hidden).to(DEVICE)
        self.predictor.train()
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=self.lr)


class Env(ABC):
    @abstractmethod
    def step_env(self, controller, action_idx):
        pass

    @abstractmethod
    def reset(self):
        pass


class RNDIntrinsicEnv(Env):
    def __init__(self, reward_module: RNDIntrinsicReward):
        self.reward_module = reward_module

    def step_env(self, controller, action_idx):
        action_str = ACTIONS[action_idx]
        event = controller.step(action_str)

        feats = self.reward_module.pre_process_rnd(event)
        reward = self.reward_module.compute_reward(feats)
        if "Move" in action_str:
            reward = reward * 1.5 if reward > 0 else reward / 1.5
        return event, reward
    
    def reset(self):
        pass


class CLIPNovelty:
    """
    Computes intrinsic reward = 1 - mean cosine similarity
    between current CLIP embedding and last N embeddings.
    """
    def __init__(self, device=DEVICE, model_name="ViT-B/32", buffer_size=EPISODE_STEPS):
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size

    @torch.no_grad()
    def compute_reward(self, frame_np):
        """
        frame_np : np.ndarray (H,W,3) uint8
        returns : float intrinsic reward
        """
        img = Image.fromarray(frame_np)
        img_t = self.preprocess(img).unsqueeze(0).to(self.device)
        emb = self.model.encode_image(img_t)
        emb = emb / emb.norm(dim=-1, keepdim=True)  # normalize to unit sphere

        if len(self.buffer) == 0:
            self.buffer.append(emb)
            return 0.0  # no past frames to compare yet

        past = torch.cat(list(self.buffer), dim=0)  # (N, D)
        sim = F.cosine_similarity(emb, past)        # (N,)
        reward = (1 - sim.mean()).item()
        self.buffer.append(emb)
        return reward
    
    def reset(self):
        self.buffer = deque(maxlen=self.buffer_size)

class ClipEnv(Env):
    def __init__(self, clip_novelty: CLIPNovelty):
        super().__init__()
        self.clip_novelty = clip_novelty
        self.positions = deque(maxlen=16)
        self.undo_count = 0
        self.last_action = ""
        
    def step_env(self, controller, action_idx):
        action_str = ACTIONS[action_idx]
        event = controller.step(action_str)
    
        reward = self.clip_novelty.compute_reward(event.frame)
        pos = np.array([event.metadata["agent"]["position"]["x"], event.metadata["agent"]["position"]["z"]])
        self.positions.append(pos)
        avg_pos = np.mean(np.stack(self.positions), axis=0)

        # --- Small bonus for moving away from recent average ---
        pos_bonus = np.linalg.norm(pos - avg_pos) / 8  # ~0–0.1 scale
        fail_penalty = 0 if event.metadata["lastActionSuccess"] else -0.1
        if "Rotate" in self.last_action and "Rotate" in action_str and self.last_action != action_str:
            self.undo_count += 1
        else:
            self.undo_count = max(0, self.undo_count - 0.5)
        undo_penalty = -0.05 * self.undo_count
        self.last_action = action_str

        return event, reward + pos_bonus + fail_penalty + undo_penalty
    
    def reset(self):
        self.clip_novelty.reset()
        self.positions = deque(maxlen=16)
        self.undo_count = 0

def save_actor_critic(actor_critic, path="actor_critic_checkpoint.pt"):
    """
    Saves all model and optimizer weights from an ActorCritic instance.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    
    checkpoint = {
        "encoder": actor_critic.actor_critic_encoder.state_dict(),
        "actor": actor_critic.actor.state_dict(),
        "critic": actor_critic.critic.state_dict(),
        "optimizer": actor_critic.optimizer.state_dict(),
    }
    
    torch.save(checkpoint, path)
    print(f"[✅] Actor-Critic checkpoint saved to {path}")

def load_actor_critic(actor_critic, path="actor_critic_checkpoint.pt", device="cuda"):
    """
    Loads model and optimizer weights into an ActorCritic instance.
    """
    checkpoint = torch.load(path, map_location=device)
    
    actor_critic.actor_critic_encoder.load_state_dict(checkpoint["encoder"])
    actor_critic.actor.load_state_dict(checkpoint["actor"])
    actor_critic.critic.load_state_dict(checkpoint["critic"])
    actor_critic.optimizer.load_state_dict(checkpoint["optimizer"])
    
    print(f"[🔁] Actor-Critic checkpoint loaded from {path}")


from abc import ABC, abstractmethod


class PPO():
    def __init__(self, ENTROPY_COEF: float):
        self.ENTROPY_COEF = ENTROPY_COEF

    def obs_from_event(self, event):
        frame = event.frame.copy()
        return transform(frame).to(DEVICE)  # (3,H,W) tensor


    @torch.no_grad()
    def act_and_value(self, obs_seq, actor_critic: ActorCritic):  # obs_seq: (1, seq_len, 3, H, W)
        # Encode each frame individually    
        feats = actor_critic.actor_critic_encoder(obs_seq)             # (B, seq_len, FEAT_DIM)

        # Actor: uses sequence
        logits = actor_critic.actor(feats, None)[-1, :]                # (B, s, NUM_ACTIONS)
        value = actor_critic.critic(feats, None)[-1, :]      


        # Critic: only use last frame (current state)
                       # (B, 1)
        return logits, value.item()

    def evaluate_batch(self, obs: torch.Tensor, actions, actor_critic: ActorCritic):
        """
        obs:     (B, S, 3, H, W)
        actions: (B*S,)
        returns:
            logps:      (B*S,)
            entropies:  (B*S,)
            values:     (B*S,)
        """
        b, s, c, h, w = obs.shape
        feats = actor_critic.actor_critic_encoder(obs)  # (B, S, D)

        # 3) Transformer over full sequence with causal mask

        causal_mask = torch.full((s, s), float('-inf'), device=feats.device)
        causal_mask = torch.triu(causal_mask, diagonal=1)

        logits = actor_critic.actor(feats, causal_mask)
        dist = torch.distributions.Categorical(logits=logits)

        logps     = dist.log_prob(actions)                      # (B*S,)
        entropies = dist.entropy()                              # (B*S,)
        # 6) Value per prefix: use last-state representation at each prefix (z at time t)
        values = actor_critic.critic(feats, causal_mask).squeeze(-1)            # (B*S,)

        return logps, entropies, values
    

    def compute_gae(self, rewards, values, dones, gamma=GAMMA, lam=GAE_LAMBDA):
        T = len(rewards)
        adv = torch.zeros(T, dtype=torch.float32)
        vals = torch.tensor(values + [0.0], dtype=torch.float32)  # bootstrap V_{T}=0 unless you pass last V
        lastgaelam = 0.0
        for t in reversed(range(T)):
            nonterminal = 1.0 - float(dones[t])
            delta = rewards[t] + gamma * vals[t+1] * nonterminal - vals[t]
            lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
            adv[t] = lastgaelam
        ret = adv + vals[:-1]
        return adv, ret
    
    def ppo_update(self, buffer: RolloutBuffer, actor_critic: ActorCritic):
        """
        PPO update compatible with:
        - Transformer actor (takes obs sequences: (B, seq_len, 3, H, W))
        - CNN critic (uses only last frame embedding)
        """
        T = len(buffer)
        c, h, w = buffer.obs[0].shape
        obs = torch.stack(buffer.obs).to(DEVICE).view(MINIBATCHES, T // MINIBATCHES, c, h, w)               # (T, seq_len, 3, H, W)
        actions = torch.tensor(buffer.actions, dtype=torch.long, device=DEVICE)  # (T,)
        old_logps = torch.tensor(buffer.logps, dtype=torch.float32, device=DEVICE)
        # Compute advantages and returns
        advantages, returns = self.compute_gae(buffer.rewards, buffer.values, buffer.dones)
        advantages = advantages.to(dtype=torch.float32, device=DEVICE)
        returns = returns.to(dtype=torch.float32, device=DEVICE)
        # Normalize advantages globally
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for epoch in range(TRAIN_EPOCHS):
            # Forward through actor + critic
            new_logp, entropy, value_pred = self.evaluate_batch(obs, actions, actor_critic)
            if epoch == TRAIN_EPOCHS - 1:
                with torch.no_grad():
                    approx_kl = (old_logps - new_logp).mean().item()
                    print("Approx KL Learned: " + str(approx_kl))
            # PPO ratio
            ratio = torch.exp(new_logp - old_logps)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - PPO_CLIP, 1.0 + PPO_CLIP) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            # Value and entropy losses
            value_loss = F.mse_loss(value_pred.reshape(*returns.shape), returns)
            entropy_bonus = entropy.mean()
            # Final total loss
            loss = policy_loss + VALUE_COEF * value_loss - self.ENTROPY_COEF * entropy_bonus
            actor_critic.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(actor_critic.actor_critic_encoder.parameters()) +
                list(actor_critic.actor.parameters()) +
                list(actor_critic.critic.parameters()),
                MAX_GRAD_NORM
            )
            actor_critic.optimizer.step()

            # Optional: print every few epochs
            if epoch % 10 == 0:
                print(f"[PPO] Epoch {epoch}: Loss={loss.item():.4f}, Policy={policy_loss.item():.4f}, Value={value_loss.item():.4f}")


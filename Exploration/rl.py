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


import torch
import torch.nn.functional as F
from collections import deque
from PIL import Image
import clip

class CLIPLongTermNovelty:
    def __init__(self, device=DEVICE, model_name="ViT-B/32",
                 ep_buf_size=EPISODE_STEPS, lt_buf_size=4096, topk=10, tau=0.75,
                 alpha=1.0, beta=0.1, scale=5.0):
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.ep_buf = deque(maxlen=ep_buf_size)                # episodic (cleared each episode)
        self.lt_buf = deque(maxlen=lt_buf_size)                # long-term (persistent)
        self.topk = topk
        self.tau = tau      # margin for penalty
        self.alpha = alpha  # weight for episodic novelty
        self.beta = beta    # weight for long-term penalty

    @torch.no_grad()
    def _encode(self, frame_np):
        img = Image.fromarray(frame_np)
        x = self.preprocess(img).unsqueeze(0).to(self.device)
        z = self.model.encode_image(x)
        z = z / z.norm(dim=-1, keepdim=True)
        return z.float()  # (1,D)

    @staticmethod
    def _topk_sim(feat, mem, k):
        mem_t = torch.cat(list(mem), dim=0)  # (N,D)
        sims = (feat @ mem_t.T).squeeze(0)  # (N,)
        k = min(k, sims.numel())
        return torch.topk(sims, k, largest=True).values.mean().item()

    def compute_reward(self, frame_np):
        feat = self._encode(frame_np)

        if len(self.ep_buf) == 0:
            self.ep_buf.append(feat.detach())
            self.lt_buf.append(feat.detach())
            return 0.0

        # Episodic novelty (encourage low similarity to current-episode frames)
        ep_sim = self._topk_sim(feat, self.ep_buf, self.topk)   # ~0.8–0.99
        ep_novelty = 1.0 - ep_sim                               # small (0.01–0.2)
        ep_term = self.alpha * (max(0, ep_novelty) ** 0.5)              # non-linear boost

        # Long-term repetition penalty (discourage re-visiting prior episodes)
        lt_sim = self._topk_sim(feat, self.lt_buf, self.topk)
        # Linear margin: no penalty below tau
        lt_pen = max(0.0, (lt_sim - self.tau) / (1 - self.tau))
        lt_term = self.beta * lt_pen

        reward = ep_term - lt_term

        self.ep_buf.append(feat.detach())
        self.lt_buf.append(feat.detach())

        return float(reward)

    def reset(self):
        # Clear only episodic memory
        self.ep_buf.clear()
        # self.lt_buf persists across episodes


class CLIPNovelty:
    """
    Intrinsic reward based on CLIP embedding novelty.
    Computes reward = β * (1 - mean(top-k cosine similarity))
    with exponential smoothing for stability.
    """
    def __init__(
        self,
        device=DEVICE,
        model_name="ViT-B/32",
        buffer_size=EPISODE_STEPS,
        topk=5,
        tau=0.95,         # smoothing factor
    ):
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.topk = topk
        self.tau = tau

    @torch.no_grad()
    def compute_reward(self, frame_np):
        """
        frame_np : np.ndarray (H,W,3) uint8
        returns : float intrinsic reward
        """
        img = Image.fromarray(frame_np)
        img_t = self.preprocess(img).unsqueeze(0).to(self.device)

        # 1) Encode and normalize embedding
        emb = self.model.encode_image(img_t)
        emb = emb / emb.norm(dim=-1, keepdim=True)

        # 2) If no past, no novelty yet
        if len(self.buffer) == 0:
            self.buffer.append(emb)
            return 0.0

        # 3) Compute cosine similarities
        past = torch.cat(list(self.buffer), dim=0)  # (N, D)
        sims = (emb @ past.T).squeeze(0)            # (N,)
        k = min(self.topk, sims.size(0))
        topk_sim = torch.topk(sims, k, largest=True).values.mean().item()

        # 4) Novelty reward (larger when less similar)
        reward = 1.0 - topk_sim

        # 6) Update buffer
        self.buffer.append(emb.detach())

        return float(reward)

    def reset(self):
        self.buffer.clear()
        self.running_reward = 0.0

class ClipEnv(Env):
    def __init__(self, clip_novelty: CLIPNovelty):
        super().__init__()
        self.clip_novelty = clip_novelty
        self.positions = deque(maxlen=16)
        
    def step_env(self, controller, action_idx):
        action_str = ACTIONS[action_idx]
        event = controller.step(action_str)
    
        reward = self.clip_novelty.compute_reward(event.frame)
        pos = np.array([event.metadata["agent"]["position"]["x"], event.metadata["agent"]["position"]["z"]])
        self.positions.append(pos)
        avg_pos = np.mean(np.stack(self.positions), axis=0)

        # --- Small bonus for moving away from recent average ---
        pos_bonus = np.linalg.norm(pos - avg_pos) / 2  # ~0–0.1 scale
        fail_penalty = 0 if event.metadata["lastActionSuccess"] else -0.2
        self.last_action = action_str

        return event, 5 * reward + pos_bonus + fail_penalty
    
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
    def act_and_value(self, obs_seq, actions_seq, actor_critic: ActorCritic):
        """
        obs_seq:     (1, seq_len, 3, H, W)
        actions_seq: (1, seq_len)  -- previous actions (use 0 for the first)
        returns:
            logits: (num_actions,)  -- policy for the *next* action
            value:  scalar          -- critic estimate for current state
        """
        # Encode visual observations
        feats = actor_critic.actor_critic_encoder(obs_seq)  # (1, seq_len, feat_dim)

        # Feed both vision + actions to actor and critic
        logits_seq = actor_critic.actor(feats, actions_seq, mask=None)  # (1*seq_len, num_actions)
        value_seq  = actor_critic.critic(feats, mask=None)              # (1*seq_len, 1)

        # Get only the *last* timestep (the most recent frame)
        logits = logits_seq[-1, :]          # (num_actions,)
        value  = value_seq[-1, :].item()    # scalar

        return logits, value

    def evaluate_batch(self, obs: torch.Tensor, actions, actor_critic: ActorCritic):
        """
        obs:     (B, S, 3, H, W)
        actions: (B*S,)  flattened actions
        returns:
            logps:      (B*S,)
            entropies:  (B*S,)
            values:     (B*S,)
        """
        b, s, c, h, w = obs.shape
        feats = actor_critic.actor_critic_encoder(obs)  # (B, S, D)

        # Causal mask (upper triangular)
        causal_mask = torch.full((s, s), float('-inf'), device=feats.device)
        causal_mask = torch.triu(causal_mask, diagonal=1)

        # ---- [1] reshape actions and shift ----
        actions = actions.view(b, s)  # reshape from (B*S,) -> (B, S)
        actions_seq = torch.cat([torch.zeros_like(actions[:, :1]), actions[:, :-1]], dim=1)  # prev actions

        # ---- [2] actor forward ----
        logits = actor_critic.actor(feats, actions_seq, causal_mask)  # (B*S, num_actions)
        dist = torch.distributions.Categorical(logits=logits)

        # ---- [3] log-probs, entropy, and critic ----
        logps = dist.log_prob(actions.view(-1))      # flatten again (B*S,)
        entropies = dist.entropy()                   # (B*S,)
        values = actor_critic.critic(feats, causal_mask).squeeze(-1)  # (B*S,)

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


class GRPO:
    def __init__(self, entropy_coef: float):
        self.entropy_coef = entropy_coef

    def obs_from_event(self, event):
        frame = event.frame.copy()
        return transform(frame).to(DEVICE)  # (3,H,W) tensor

    @torch.no_grad()
    def act(self, obs_seq, actions_seq, actor_critic: ActorCritic):
        """
        obs_seq: (1, seq_len, 3, H, W)
        actions_seq: (1, seq_len)
        returns: logits (num_actions,)
        """
        feats = actor_critic.actor_critic_encoder(obs_seq)
        logits_seq = actor_critic.actor(feats, actions_seq, mask=None)
        logits = logits_seq[-1, :]  # only latest timestep
        return logits

    def evaluate_batch(self, obs, actions, actor_critic):
        """
        obs:     (B, S, 3, H, W)
        actions: (B*S,)
        returns: logps, entropies
        """
        b, s, c, h, w = obs.shape
        feats = actor_critic.actor_critic_encoder(obs)  # (B, S, D)

        # Causal mask for autoregressive transformer
        causal_mask = torch.full((s, s), float('-inf'), device=feats.device)
        causal_mask = torch.triu(causal_mask, diagonal=1)

        # Shifted actions (previous actions as input)
        actions = actions.view(b, s)
        actions_seq = torch.cat(
            [torch.zeros_like(actions[:, :1]), actions[:, :-1]], dim=1
        )

        logits = actor_critic.actor(feats, actions_seq, causal_mask)
        dist = torch.distributions.Categorical(logits=logits)
        logps = dist.log_prob(actions.view(-1))
        entropies = dist.entropy()

        return logps, entropies

    def compute_advantages(self, rewards, gamma=GAMMA):
        """
        Compute discounted returns directly (no critic).
        """
        T = len(rewards)
        returns = torch.zeros(T, dtype=torch.float32)
        G = 0.0
        for t in reversed(range(T)):
            G = rewards[t] + gamma * G
            returns[t] = G
        # Normalize returns → used as advantage
        advantages = (returns - returns.mean()) / (returns.std() + 1e-8)
        return advantages, returns

    def grpo_update(self, buffer: RolloutBuffer, actor_critic: ActorCritic):
        """
        GRPO update:
        - No critic, no value loss
        - Uses normalized discounted returns as advantages
        """
        T = len(buffer)
        c, h, w = buffer.obs[0].shape
        obs = torch.stack(buffer.obs).to(DEVICE).view(MINIBATCHES, T // MINIBATCHES, c, h, w)
        actions = torch.tensor(buffer.actions, dtype=torch.long, device=DEVICE)
        old_logps = torch.tensor(buffer.logps, dtype=torch.float32, device=DEVICE)

        # Compute advantages purely from rewards
        advantages, returns = self.compute_advantages(buffer.rewards)
        advantages = advantages.to(DEVICE)

        for epoch in range(TRAIN_EPOCHS):
            new_logp, entropy = self.evaluate_batch(obs, actions, actor_critic)

            ratio = torch.exp(new_logp - old_logps)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - PPO_CLIP, 1.0 + PPO_CLIP) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            entropy_bonus = entropy.mean()

            # Total GRPO loss
            loss = policy_loss - self.entropy_coef * entropy_bonus

            actor_critic.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(actor_critic.actor_critic_encoder.parameters()) +
                list(actor_critic.actor.parameters()),
                MAX_GRAD_NORM
            )
            actor_critic.optimizer.step()

            if epoch % 10 == 0:
                print(f"[GRPO] Epoch {epoch}: Loss={loss.item():.4f}, Policy={policy_loss.item():.4f}")

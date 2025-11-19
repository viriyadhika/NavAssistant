import os
from abc import ABC, abstractmethod
from collections import deque
from typing import Callable

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import imageio
from PIL import Image

import clip

from cons import (
    NUM_ACTIONS,
    FEAT_DIM,
    DEVICE,
    LR,
    transform,
    ACTIONS,
    GAMMA,
    GAE_LAMBDA,
    TRAIN_EPOCHS,
    MINIBATCHES,
    PPO_CLIP,
    VALUE_COEF,
    MAX_GRAD_NORM,
    EPISODE_STEPS,
)

# ---------------------------------------------------------
#  Rollout Buffer
# ---------------------------------------------------------
class RolloutBuffer:
    def __init__(self):
        self.encoded_rgb = []
        self.encoded_depth = []
        self.actions, self.logps = [], []
        self.rewards, self.values, self.dones = [], [], []

    def add(self, rgb_embed, depth_embed, action_idx, logp, reward, value, done):
        self.encoded_rgb.append(rgb_embed.cpu())
        self.encoded_depth.append(depth_embed.cpu())
        self.actions.append(int(action_idx))
        self.logps.append(float(logp))
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.dones.append(bool(done))

    def __len__(self):
        return len(self.rewards)


# ---------------------------------------------------------
#  Actor-Critic Container
# ---------------------------------------------------------
class ActorCritic:
    def __init__(self,
                 rgb_encoder: nn.Module,
                 depth_encoder: nn.Module,
                 actor: nn.Module,
                 critic: nn.Module):
        self.rgb_encoder = rgb_encoder.to(DEVICE)
        self.depth_encoder = depth_encoder.to(DEVICE)
        self.actor = actor.to(DEVICE)
        self.critic = critic.to(DEVICE)

        self.optimizer = torch.optim.AdamW(
            list(self.rgb_encoder.parameters()) +
            list(self.depth_encoder.parameters()) +
            list(self.actor.parameters()) +
            list(self.critic.parameters()),
            lr=LR
        )


def save_actor_critic(actor_critic: ActorCritic, path: str = "actor_critic_checkpoint.pt"):
    """
    Saves all model and optimizer weights from an ActorCritic instance.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    checkpoint = {
        "rgb_encoder": actor_critic.rgb_encoder.state_dict(),
        "depth_encoder": actor_critic.depth_encoder.state_dict(),
        "actor": actor_critic.actor.state_dict(),
        "critic": actor_critic.critic.state_dict(),
        "optimizer": actor_critic.optimizer.state_dict(),
    }

    torch.save(checkpoint, path)
    print(f"[✅] Actor-Critic checkpoint saved to {path}")


def load_actor_critic(actor_critic: ActorCritic, path: str = "actor_critic_checkpoint.pt", device: str = DEVICE):
    """
    Loads model and optimizer weights into an ActorCritic instance.
    """
    checkpoint = torch.load(path, map_location=device)

    actor_critic.rgb_encoder.load_state_dict(checkpoint["rgb_encoder"])
    actor_critic.depth_encoder.load_state_dict(checkpoint["depth_encoder"])
    actor_critic.actor.load_state_dict(checkpoint["actor"])
    actor_critic.critic.load_state_dict(checkpoint["critic"])
    actor_critic.optimizer.load_state_dict(checkpoint["optimizer"])

    print(f"[🔁] Actor-Critic checkpoint loaded from {path}")


# ---------------------------------------------------------
#  CLIP Novelty (Intrinsic Reward)
# ---------------------------------------------------------
class CLIPNovelty:
    """
    Intrinsic reward based on CLIP embedding novelty.
    Computes reward = (1 - mean(top-k cosine similarity))
    with exponential smoothing for stability.
    """
    def __init__(
        self,
        device=DEVICE,
        model_name="ViT-B/32",
        buffer_size=EPISODE_STEPS,
        topk=5,
        tau=0.95,
    ):
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.topk = topk
        self.tau = tau
        self.running_reward = 0.0  # EMA baseline

    @torch.no_grad()
    def compute_reward(self, frame_np):
        """
        frame_np : np.ndarray (H,W,3) uint8
        returns : float intrinsic reward
        """
        img = Image.fromarray(frame_np)
        img_t = self.preprocess(img).unsqueeze(0).to(self.device)

        emb = self.model.encode_image(img_t)
        emb = emb / emb.norm(dim=-1, keepdim=True)

        if len(self.buffer) == 0:
            self.buffer.append(emb)
            return 0.0

        past = torch.cat(list(self.buffer), dim=0)  # (N, D)
        sims = (emb @ past.T).squeeze(0)            # (N,)
        k = min(self.topk, sims.size(0))
        topk_sim = torch.topk(sims, k, largest=True).values.mean().item()

        reward_raw = 1.0 - topk_sim

        self.running_reward = self.tau * self.running_reward + (1 - self.tau) * reward_raw
        reward = reward_raw - self.running_reward

        self.buffer.append(emb.detach())
        return float(reward)

    def reset(self):
        self.buffer.clear()
        self.running_reward = 0.0


# ---------------------------------------------------------
#  Environment Abstraction & Variants
# ---------------------------------------------------------
class Env(ABC):
    @abstractmethod
    def step_env(self, controller, action_idx):
        pass

    @abstractmethod
    def reset(self):
        pass


class ClipEnv(Env):
    def __init__(self, clip_novelty: CLIPNovelty):
        super().__init__()
        self.clip_novelty = clip_novelty
        self.positions = deque(maxlen=32)

    def step_env(self, controller, action_idx):
        action_str = ACTIONS[action_idx]
        event = controller.step(action_str)

        reward = self.clip_novelty.compute_reward(event.frame)
        pos = np.array([
            event.metadata["agent"]["position"]["x"],
            event.metadata["agent"]["position"]["z"]
        ])
        self.positions.append(pos)
        avg_pos = np.mean(np.stack(self.positions), axis=0)

        pos_bonus = np.linalg.norm(pos - avg_pos) / 2.0  # ~0–0.1 scale
        self.last_action = action_str

        return event, 3 * reward + pos_bonus

    def reset(self):
        self.clip_novelty.reset()
        self.positions = deque(maxlen=32)


class ClipEnvNoPenalty(Env):
    def __init__(self, clip_novelty: CLIPNovelty):
        super().__init__()
        self.clip_novelty = clip_novelty
        self.positions = deque(maxlen=32)

    def step_env(self, controller, action_idx):
        action_str = ACTIONS[action_idx]
        event = controller.step(action_str)

        reward = self.clip_novelty.compute_reward(event.frame)
        pos = np.array([
            event.metadata["agent"]["position"]["x"],
            event.metadata["agent"]["position"]["z"]
        ])
        self.positions.append(pos)
        avg_pos = np.mean(np.stack(self.positions), axis=0)

        pos_bonus = np.linalg.norm(pos - avg_pos) / 2.0
        self.last_action = action_str

        return event, 3 * reward + pos_bonus

    def reset(self):
        self.clip_novelty.reset()
        self.positions = deque(maxlen=32)


class ClipEnvNoCuriosity(Env):
    def __init__(self, clip_novelty: CLIPNovelty):
        super().__init__()
        self.clip_novelty = clip_novelty
        self.positions = deque(maxlen=32)

    def step_env(self, controller, action_idx):
        action_str = ACTIONS[action_idx]
        event = controller.step(action_str)

        pos = np.array([
            event.metadata["agent"]["position"]["x"],
            event.metadata["agent"]["position"]["z"]
        ])
        self.positions.append(pos)
        avg_pos = np.mean(np.stack(self.positions), axis=0)

        pos_bonus = np.linalg.norm(pos - avg_pos) / 2.0
        self.last_action = action_str
        fail_penalty = 0 if event.metadata["lastActionSuccess"] else -0.2

        return event, 2 * (pos_bonus + fail_penalty)

    def reset(self):
        self.clip_novelty.reset()
        self.positions = deque(maxlen=32)


# ---------------------------------------------------------
#  PPO
# ---------------------------------------------------------
class PPO:
    def __init__(self, ENTROPY_COEF: float):
        self.ENTROPY_COEF = ENTROPY_COEF

    def obs_from_event(self, event):
        frame = event.frame.copy()
        return transform(frame).to(DEVICE)  # (3,H,W) tensor

    @torch.no_grad()
    def act_and_value(self, feats, actions_seq, actor_critic: ActorCritic):
        """
        feats:        (1, S, D_fused)  fused RGB+Depth embeddings
        actions_seq:  (1, S)
        returns:
            logits: (num_actions,)
            value:  scalar
        """
        W = actor_critic.actor.window
        _, S, D = feats.shape

        if S >= W:
            feats_win = feats[:, -W:, :]         # (1, W, D)
            acts_win = actions_seq[:, -W:]       # (1, W)
        else:
            pad = W - S
            feats_pad = feats[:, :1, :].repeat(1, pad, 1)
            acts_pad = actions_seq[:, :1].repeat(1, pad)

            feats_win = torch.cat([feats_pad, feats], dim=1)  # (1, W, D)
            acts_win = torch.cat([acts_pad, actions_seq], dim=1)

        logits_seq = actor_critic.actor(feats_win, acts_win, mask=None)  # (W, num_actions)
        value_seq = actor_critic.critic(feats_win, mask=None)            # (W, 1)

        logits = logits_seq[-1]          # (num_actions,)
        value = value_seq[-1].item()     # scalar

        return logits, value

    def evaluate_batch(self, feats: torch.Tensor, actions, actor_critic: ActorCritic):
        """
        feats:   (B, S, D) fused RGB+Depth embeddings
        actions: (B*S,)   flattened actions

        returns:
            logps:      (B*S,)
            entropies:  (B*S,)
            values:     (B*S,)
        """
        b, s, d = feats.shape

        # causal mask (upper triangular) if you ever want to use it
        causal_mask = torch.full((s, s), float('-inf'), device=feats.device)
        causal_mask = torch.triu(causal_mask, diagonal=1)

        actions = actions.view(b, s)  # (B, S)
        actions_seq = torch.cat(
            [torch.zeros_like(actions[:, :1]), actions[:, :-1]], dim=1
        )  # previous actions

        logits = actor_critic.actor(feats, actions_seq, causal_mask)  # (B*S, num_actions)
        dist = torch.distributions.Categorical(logits=logits)

        logps = dist.log_prob(actions.view(-1))    # (B*S,)
        entropies = dist.entropy()                 # (B*S,)
        values = actor_critic.critic(feats, causal_mask).squeeze(-1)  # (B*S,)

        return logps, entropies, values

    def compute_gae(self, rewards, values, dones, gamma=GAMMA, lam=GAE_LAMBDA):
        T = len(rewards)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        adv = torch.zeros(T, dtype=torch.float32)
        vals = torch.tensor(values + [0.0], dtype=torch.float32)  # bootstrap V_T = 0
        lastgaelam = 0.0

        for t in reversed(range(T)):
            nonterminal = 1.0 - float(dones[t])
            delta = rewards[t] + gamma * vals[t + 1] * nonterminal - vals[t]
            lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
            adv[t] = lastgaelam

        ret = adv + vals[:-1]
        return adv, ret

    def ppo_update(self, buffer: RolloutBuffer, actor_critic: ActorCritic):
        """
        PPO update using fused RGB+Depth features stored in buffer.
        """
        T = len(buffer)

        rgb = torch.stack(buffer.encoded_rgb).to(DEVICE)       # (T, D)
        depth = torch.stack(buffer.encoded_depth).to(DEVICE)   # (T, D)

        # simple fusion: addition (same dimensionality)
        fused = torch.cat([rgb, depth], dim=-1)              # (T, D)
        d = fused.shape[-1]

        fused = fused.view(MINIBATCHES, T // MINIBATCHES, d)   # (B, S, D)
        actions = torch.tensor(buffer.actions, dtype=torch.long, device=DEVICE)
        old_logps = torch.tensor(buffer.logps, dtype=torch.float32, device=DEVICE)

        advantages, returns = self.compute_gae(buffer.rewards, buffer.values, buffer.dones)
        advantages = advantages.to(dtype=torch.float32, device=DEVICE)
        returns = returns.to(dtype=torch.float32, device=DEVICE)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for epoch in range(TRAIN_EPOCHS):
            new_logp, entropy, value_pred = self.evaluate_batch(fused, actions, actor_critic)

            if epoch == TRAIN_EPOCHS - 1:
                with torch.no_grad():
                    approx_kl = (old_logps - new_logp).mean().item()
                    print("Approx KL Learned:", approx_kl)

            ratio = torch.exp(new_logp - old_logps)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - PPO_CLIP, 1.0 + PPO_CLIP) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(value_pred.reshape(*returns.shape), returns)
            entropy_bonus = entropy.mean()

            loss = policy_loss + VALUE_COEF * value_loss - self.ENTROPY_COEF * entropy_bonus

            actor_critic.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(actor_critic.rgb_encoder.parameters()) +
                list(actor_critic.depth_encoder.parameters()) +
                list(actor_critic.actor.parameters()) +
                list(actor_critic.critic.parameters()),
                MAX_GRAD_NORM
            )
            actor_critic.optimizer.step()

            if epoch % 10 == 0:
                print(
                    f"[PPO] Epoch {epoch}: "
                    f"Loss={loss.item():.4f}, "
                    f"Policy={policy_loss.item():.4f}, "
                    f"Value={value_loss.item():.4f}"
                )


# ---------------------------------------------------------
#  Teleport helper
# ---------------------------------------------------------
def teleport(controller, target=None):
    event = controller.step("GetReachablePositions")
    reachable_positions = event.metadata["actionReturn"]

    if target is None:
        target = np.random.choice(reachable_positions)

    event = controller.step(
        action="TeleportFull",
        x=target["x"],
        y=target["y"],
        z=target["z"],
        rotation={"x": 0, "y": 0, "z": 0},
        horizon=0,
        standing=True
    )

    return event


# ---------------------------------------------------------
#  Inference / Visualization
# ---------------------------------------------------------
def inference(
        get_distribution,
        controller,
        ppo: PPO,
        init_position: dict,
        env: Env,
        actor_critic: ActorCritic,
        plot=True,
        n_steps=256,
        n_rows=32
    ):

    positions = []

    if plot:
        plt.figure(figsize=(n_steps // n_rows * 2, n_rows * 2))

    # reset env
    event = teleport(controller, init_position)

    episode_seq = deque(maxlen=EPISODE_STEPS)
    actions_seq = deque(maxlen=EPISODE_STEPS)
    raw_obs = []

    for t in range(1, n_steps + 1):

        # Track x,z positions
        positions.append([
            event.metadata["agent"]["position"]["x"],
            event.metadata["agent"]["position"]["z"],
        ])

        # ------------------------------
        # 1. Encode RGB
        # ------------------------------
        rgb_t = ppo.obs_from_event(event)              # (3,H,W)
        rgb_t = rgb_t.unsqueeze(0).unsqueeze(0)        # (1,1,3,H,W)
        with torch.no_grad():
            rgb_embed = actor_critic.rgb_encoder(rgb_t).squeeze(0).squeeze(0)

        # ------------------------------
        # 2. Encode DEPTH
        # ------------------------------
        depth_np = event.depth_frame              # numpy (H,W)
        depth_t = torch.from_numpy(depth_np.copy()).float()
        depth_t = depth_t.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1,1,1,H,W)
        depth_t = depth_t.to(DEVICE) / 10

        with torch.no_grad():
            depth_embed = actor_critic.depth_encoder(depth_t).squeeze(0).squeeze(0)

        # ------------------------------
        # 3. Fuse (RGB + Depth)
        # ------------------------------
        fused_embed = torch.cat([rgb_embed, depth_embed], dim=-1)

        raw_obs.append(rgb_t)

        # ------------------------------
        # 4. Build observation sequence
        # ------------------------------
        episode_seq.append(fused_embed)

        obs_seq = torch.stack(list(episode_seq), dim=0).unsqueeze(0).to(DEVICE)
        #         (1, S, 2D)

        # ------------------------------
        # 5. Build action sequence
        # ------------------------------
        if len(actions_seq) == 0:
            # warm start
            actions_seq.append(torch.randint(0, NUM_ACTIONS, ()).item())

        actions_tensor = torch.tensor(list(actions_seq), dtype=torch.long, device=DEVICE)
        actions_tensor = actions_tensor.unsqueeze(0)  # (1,S)

        # ------------------------------
        # 6. Actor forward
        # ------------------------------
        dist = get_distribution(ppo, obs_seq, actions_tensor, actor_critic)

        action_idx = dist.sample().item()
        logp = dist.log_prob(torch.tensor(action_idx, device=DEVICE)).item()

        # ------------------------------
        # 7. Step environment
        # ------------------------------
        event, reward = env.step_env(controller, action_idx)

        actions_seq.append(action_idx)

        # ------------------------------
        # 8. Plotting
        # ------------------------------
        if plot:
            plt.subplot(n_rows, n_steps // n_rows, t)
            probs = torch.exp(dist.log_prob(torch.tensor([0,1,2], device=DEVICE))).cpu().numpy()
            plt.title(f"act={ACTIONS[action_idx]}\nr={reward:.2f}\np={probs}", fontsize=6)
            plt.axis(False)
            plt.imshow(event.frame)

    if plot:
        plt.tight_layout()
        plt.show()

        # ---- Plot 2D trajectory ----
        positions = np.array(positions)
        plt.figure(figsize=(4, 4))
        plt.plot(positions[:, 0], positions[:, 1], "-o", markersize=3)
        plt.xlabel("x")
        plt.ylabel("z")
        plt.title("Agent trajectory")
        plt.grid(True)
        plt.show()

    return raw_obs


def inference_video_mp4(
    get_distribution,
    controller,
    ppo: PPO,
    init_position: dict[str, float],
    env: Env,
    actor_critic: ActorCritic,
    video_path="rollout.mp4",
    fps=10,
    n_steps=512,
):
    episode_seq = deque(maxlen=EPISODE_STEPS)
    actions_seq = deque(maxlen=EPISODE_STEPS)

    writer = imageio.get_writer(video_path, fps=fps)

    # reset env
    event = teleport(controller, init_position)
    positions = []

    for t in range(1, n_steps + 1):

        # Write RGB frame to video
        writer.append_data(event.frame)

        # Track position
        positions.append([
            event.metadata["agent"]["position"]["x"],
            event.metadata["agent"]["position"]["z"],
        ])

        # -------------------------------------
        # 1. Encode RGB
        # -------------------------------------
        with torch.no_grad():
            rgb_t = ppo.obs_from_event(event)                   # (3,H,W)
            rgb_t = rgb_t.unsqueeze(0).unsqueeze(0)             # (1,1,3,H,W)
            rgb_enc = actor_critic.rgb_encoder(rgb_t).squeeze(0).squeeze(0)

        # -------------------------------------
        # 2. Encode Depth
        # -------------------------------------
        depth_np = event.depth_frame                           # numpy (H,W)
        depth_t = torch.from_numpy(depth_np.copy()).float()
        depth_t = depth_t.unsqueeze(0).unsqueeze(0).unsqueeze(0)   # (1,1,1,H,W)
        depth_t = depth_t.to(DEVICE) / 10

        with torch.no_grad():
            depth_enc = actor_critic.depth_encoder(depth_t).squeeze(0).squeeze(0)

        # -------------------------------------
        # 3. Fuse embeddings
        # -------------------------------------
        fused = torch.cat([rgb_enc, depth_enc], dim=-1)         # (2D,)

        # -------------------------------------
        # 4. Build observation sequence
        # -------------------------------------
        episode_seq.append(fused)

        obs_seq = torch.stack(list(episode_seq), dim=0)         # (S,2D)
        obs_seq = obs_seq.unsqueeze(0).to(DEVICE)               # (1,S,2D)

        # -------------------------------------
        # 5. Action sequence
        # -------------------------------------
        if len(actions_seq) == 0:
            actions_seq.append(torch.randint(0, NUM_ACTIONS, ()).item())

        actions_tensor = torch.tensor(
            list(actions_seq), dtype=torch.long, device=DEVICE
        ).unsqueeze(0)                                          # (1,S)

        # -------------------------------------
        # 6. Query policy
        # -------------------------------------
        dist = get_distribution(ppo, obs_seq, actions_tensor, actor_critic)
        action_idx = dist.sample().item()

        # -------------------------------------
        # 7. Step environment
        # -------------------------------------
        event, reward = env.step_env(controller, action_idx)

        actions_seq.append(action_idx)

    # -------------------------------------
    # Finalize
    # -------------------------------------
    writer.close()
    print(f"[🎞️] Saved video to {video_path}")

    return positions

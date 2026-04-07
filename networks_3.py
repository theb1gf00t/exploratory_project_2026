"""
networks_3.py  (scaled — daily sortie observation space)

Scaled-up neural networks for MAPPO UAV crop monitoring.

SectorAttentionActor  (~334K params per actor)
  Sector tokens (risk weight + status + coordinates) are embedded and
  processed through a 2-layer Transformer encoder. The attended
  representation is combined with own state and other-UAV positions via an MLP.

  Architecture:
    sector_embed   : Linear(4 → 128)  — (risk_w, status_norm, row_norm, col_norm)
    transformer    : 2× TransformerEncoderLayer(d=128, heads=4, ffn=256)
    global_embed   : Linear(15 → 128) — own(9) + other_uav_deltas(6)
    action_mean    : Linear(256→256)→ReLU→Linear(256→2)→Tanh  — mean [vx, vy]
    action_log_std : nn.Parameter(1, 2)  — learnable log std

CriticNetwork  (~606K params)
  Large MLP on the joint observation of all UAVs concatenated.

  Architecture:
    Linear(JOINT_SIZE→512)→ReLU→Linear(512→256)→ReLU→Linear(256→128)→ReLU→Linear(128→1)
    Output: scalar state value V(s)

Total model size: ~1.94M parameters

Fix 9: imports from uav_env_3 (double-crash-penalty fix, W_UNKNOWN_FLOOR=0.1,
        proximity tie-breaking, PBRS_SCALE=2.0).
"""

import torch
import torch.nn as nn
from torch.distributions import Normal

from uav_env_3 import N_SECTORS, N_UAVS, OBS_SIZE, JOINT_SIZE, GRID_ROWS, GRID_COLS

# Observation slice indices (must match uav_env_3._get_obs)
_OWN_END    = 9                          # Item 46
_RISK_END   = 9 + N_SECTORS             # 109  — Item 47
_STATUS_END = 9 + N_SECTORS * 2         # 209  — Item 48
# [209 : OBS_SIZE] = other-UAV positions  (6 values for 4 UAVs)  — Item 49

_GLOBAL_DIM = _OWN_END + (N_UAVS - 1) * 2   # 9 + 6 = 15  — Items 50, 51
_D_MODEL    = 128
_N_HEADS    = 4
_N_LAYERS   = 2
_FFN_DIM    = 256


class SectorAttentionActor(nn.Module):
    """
    Attention-based actor for one UAV.

    Forward input  : obs tensor of shape (..., OBS_SIZE)
    Forward output : Normal distribution over continuous velocity actions
    """

    def __init__(self):
        super().__init__()

        self.sector_embed = nn.Linear(4, _D_MODEL)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=_D_MODEL, nhead=_N_HEADS, dim_feedforward=_FFN_DIM,
            dropout=0.0, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=_N_LAYERS)
        self.global_embed = nn.Linear(_GLOBAL_DIM, _D_MODEL)

        self.action_mean = nn.Sequential(
            nn.Linear(_D_MODEL * 2, _FFN_DIM),
            nn.ReLU(),
            nn.Linear(_FFN_DIM, 2),
            nn.Tanh()
        )

        self.action_log_std = nn.Parameter(torch.zeros(1, 2))

        self.register_buffer("sector_coords", self._build_coords())

    def _build_coords(self):
        coords = torch.zeros(N_SECTORS, 2)
        for sid in range(N_SECTORS):
            coords[sid, 0] = (sid // GRID_COLS) / (GRID_ROWS - 1)  # row_norm
            coords[sid, 1] = (sid %  GRID_COLS) / (GRID_COLS - 1)  # col_norm
        return coords

    def forward(self, obs):
        """
        obs : (..., OBS_SIZE)
        returns Normal distribution (Eq. 15, 19)
        """
        own_feats    = obs[..., :_OWN_END]                     # (..., 9)
        risk_w       = obs[..., _OWN_END:_RISK_END]            # (..., 100)
        status_n     = obs[..., _RISK_END:_STATUS_END]         # (..., 100)
        other_uav    = obs[..., _STATUS_END:]                  # (..., 6)

        # Build base sector tokens: (..., N_SECTORS, 2)
        sector_feats = torch.stack([risk_w, status_n], dim=-1)

        # Handle arbitrary leading batch dims by flattening to (B, N, 2)
        leading      = sector_feats.shape[:-2]
        B            = 1
        for d in leading:
            B *= d
        sector_feats = sector_feats.view(B, N_SECTORS, 2)

        # Attach 2D physical coordinates to every token
        coords_expanded = self.sector_coords.unsqueeze(0).expand(B, N_SECTORS, 2)
        full_sector_feats = torch.cat([sector_feats, coords_expanded], dim=-1)

        # Embed the 4-feature tokens directly
        sector_emb   = self.sector_embed(full_sector_feats)            # (B, 100, 128)

        # Transformer Processing
        sector_ctx   = self.transformer(sector_emb)                    # (B, 100, 128)
        sector_pool  = sector_ctx.mean(dim=1)                          # (B, 128)

        # Global features
        global_feats = torch.cat(
            [own_feats.view(B, _OWN_END),
             other_uav.view(B, (N_UAVS - 1) * 2)], dim=-1
        )                                                              # (B, 15)  — Item 52
        global_emb   = self.global_embed(global_feats)                 # (B, 128)

        # Combine and produce Gaussian parameters (Eq. 15)
        combined     = torch.cat([sector_pool, global_emb], dim=-1)    # (B, 256)

        mu = self.action_mean(combined)                                # (B, 2)
        std = torch.exp(self.action_log_std).expand_as(mu)             # (B, 2)

        # Restore leading dims if necessary
        if leading:
            mu  = mu.view(*leading, 2)
            std = std.view(*leading, 2)

        return Normal(mu, std)

    def get_action(self, obs):
        """Sample a continuous velocity vector. obs: (OBS_SIZE,) or (1, OBS_SIZE)."""
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        dist = self.forward(obs)
        action = dist.sample()
        return action.squeeze(0).cpu().numpy(), dist.log_prob(action).sum(dim=-1)

    def get_log_prob_entropy(self, obs, actions):
        """Compute log-probs and mean entropy for continuous actions."""
        dist = self.forward(obs)
        log_p = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().mean()
        return log_p, entropy


class CriticNetwork(nn.Module):
    """
    Centralised critic — takes the joint observation of all UAVs.

    Input  : (batch, JOINT_SIZE=860)
    Output : (batch, 1)  — state value V(s)
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(JOINT_SIZE, 512), nn.ReLU(),
            nn.Linear(512,        256), nn.ReLU(),
            nn.Linear(256,        128), nn.ReLU(),
            nn.Linear(128,          1),
        )

    def forward(self, joint_obs):
        return self.net(joint_obs)


# ── Convenience: param count ──────────────────────────────────────────────────

def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    actor  = SectorAttentionActor()
    critic = CriticNetwork()

    actor_p  = count_params(actor)
    critic_p = count_params(critic)
    total_p  = actor_p * N_UAVS + critic_p

    print(f"SectorAttentionActor : {actor_p:>10,} params  (×{N_UAVS} UAVs = {actor_p*N_UAVS:,})")
    print(f"CriticNetwork        : {critic_p:>10,} params")
    print(f"Total                : {total_p:>10,} params")

    # Smoke test
    import os, sys
    obs_batch = torch.randn(30, OBS_SIZE)
    dist      = actor(obs_batch)
    print(f"\nActor output (batch=30) mean shape: {dist.loc.shape}  ✓")

    joint_batch = torch.randn(30, JOINT_SIZE)
    val         = critic(joint_batch)
    print(f"Critic output (batch=30): {val.shape}  ✓")

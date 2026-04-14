"""
networks_4.py

Attention-based actor / centralized critic for uav_env_4.py.
Updates observation slicing for the 5 environmental variables added to
the per-UAV observation vector.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal

from uav_env_4 import (
    ENV_OBS_DIM,
    GRID_COLS,
    GRID_ROWS,
    JOINT_SIZE,
    N_SECTORS,
    N_UAVS,
    OBS_SIZE,
)

_OWN_END = 9
_ENV_END = 9 + ENV_OBS_DIM
_RISK_END = _ENV_END + N_SECTORS
_STATUS_END = _RISK_END + N_SECTORS

_GLOBAL_DIM = _OWN_END + ENV_OBS_DIM + (N_UAVS - 1) * 2
_D_MODEL = 128
_N_HEADS = 4
_N_LAYERS = 2
_FFN_DIM = 256


class SectorAttentionActor(nn.Module):
    def __init__(self):
        super().__init__()

        self.sector_embed = nn.Linear(4, _D_MODEL)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=_D_MODEL,
            nhead=_N_HEADS,
            dim_feedforward=_FFN_DIM,
            dropout=0.0,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=_N_LAYERS)
        self.global_embed = nn.Linear(_GLOBAL_DIM, _D_MODEL)

        self.action_mean = nn.Sequential(
            nn.Linear(_D_MODEL * 2, _FFN_DIM),
            nn.ReLU(),
            nn.Linear(_FFN_DIM, 2),
            nn.Tanh(),
        )
        self.action_log_std = nn.Parameter(torch.zeros(1, 2))

        self.register_buffer("sector_coords", self._build_coords())

    def _build_coords(self):
        coords = torch.zeros(N_SECTORS, 2)
        for sid in range(N_SECTORS):
            coords[sid, 0] = (sid // GRID_COLS) / (GRID_ROWS - 1)
            coords[sid, 1] = (sid % GRID_COLS) / (GRID_COLS - 1)
        return coords

    def forward(self, obs):
        own_feats = obs[..., :_OWN_END]
        env_feats = obs[..., _OWN_END:_ENV_END]
        risk_w = obs[..., _ENV_END:_RISK_END]
        status_n = obs[..., _RISK_END:_STATUS_END]
        other_uav = obs[..., _STATUS_END:]

        sector_feats = torch.stack([risk_w, status_n], dim=-1)

        leading = sector_feats.shape[:-2]
        batch = 1
        for dim in leading:
            batch *= dim
        sector_feats = sector_feats.view(batch, N_SECTORS, 2)

        coords_expanded = self.sector_coords.unsqueeze(0).expand(batch, N_SECTORS, 2)
        full_sector_feats = torch.cat([sector_feats, coords_expanded], dim=-1)

        sector_emb = self.sector_embed(full_sector_feats)
        sector_ctx = self.transformer(sector_emb)
        sector_pool = sector_ctx.mean(dim=1)

        global_feats = torch.cat(
            [
                own_feats.view(batch, _OWN_END),
                env_feats.view(batch, ENV_OBS_DIM),
                other_uav.view(batch, (N_UAVS - 1) * 2),
            ],
            dim=-1,
        )
        global_emb = self.global_embed(global_feats)

        combined = torch.cat([sector_pool, global_emb], dim=-1)
        mu = self.action_mean(combined)
        std = torch.exp(self.action_log_std).expand_as(mu)

        if leading:
            mu = mu.view(*leading, 2)
            std = std.view(*leading, 2)

        return Normal(mu, std)

    def get_action(self, obs):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        dist = self.forward(obs)
        action = dist.sample()
        return action.squeeze(0).cpu().numpy(), dist.log_prob(action).sum(dim=-1)

    def get_log_prob_entropy(self, obs, actions):
        dist = self.forward(obs)
        log_p = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().mean()
        return log_p, entropy


class CriticNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(JOINT_SIZE, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, joint_obs):
        return self.net(joint_obs)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    actor = SectorAttentionActor()
    critic = CriticNetwork()

    print(f"Actor params : {count_params(actor):,}")
    print(f"Critic params: {count_params(critic):,}")

    obs_batch = torch.randn(30, OBS_SIZE)
    dist = actor(obs_batch)
    print(f"Actor output mean shape : {dist.loc.shape}")

    joint_batch = torch.randn(30, JOINT_SIZE)
    value = critic(joint_batch)
    print(f"Critic output shape     : {value.shape}")

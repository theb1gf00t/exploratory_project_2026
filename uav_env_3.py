"""
uav_env_3.py  (scaled — daily sortie architecture, fix round 2)

Gym-style environment for the scaled-up Multi-UAV Active Sensing system.
10×10 grid, 4 UAVs, 72-day episodes with intra-day sortie loops.

Each day consists of DAILY_STEPS_MAX=40 inner steps. UAVs launch from
their home Access Points (APs), explore/diagnose, and must return before
energy runs out or the day ends. Disease advances once per day.

Implements the same core equations as the original:
  Eq. 2  → Detection history H[k,t]
  Eq. 4  → Risk potential Omega[k,t]
  Eq. 5  → Dynamic risk weight w[k,t]
  Eq. 6  → Global risk density field
  Eq. 9  → Inter-UAV repulsion
  Eq. 11 → PBRS composite reward (explore / return mode)

Observation vector per UAV (215 values):
  [row_norm, col_norm, energy_norm, vx, vy,          (5)  own state
   dist_to_AP_row_norm, dist_to_AP_col_norm,         (2)  AP-relative
   survival_ratio_clipped, daily_step_norm,           (2)  sortie phase
   risk_w[0..99],                                   (100) risk weights
   status_norm[0..99],                              (100) 0=H, 0.5=I, 1.0=?
   Δrow_j/Δcol_j for each other UAV]                 (6) 3 others × 2

Status codes:
  0 = healthy  (UAV diagnosed)
  1 = infected (UAV diagnosed)
  2 = unknown  (not yet visited)

Changes from uav_env_2.py
─────────────────────────
  Fix 1 — Double CRASH_PENALTY removed from _compute_reward.
           The training loop applies CRASH_PENALTY via info['newly_crashed'].
           _compute_reward previously also applied it, causing energy-depletion
           crashes to be penalised -200 instead of -100, inconsistent with
           time-out crashes which were only penalised -100 by the training loop.

  Fix 2 — W_UNKNOWN_FLOOR: 0.0 → 0.1
           At episode start H=0 everywhere and no infected sectors are known,
           so omega=0 for all unknowns.  With floor=0.0 np.argmax on all-zeros
           always returns index 0 (sector_0), sending every UAV toward the same
           corner.  Floor=0.1 prevents the degenerate argmax while keeping the
           risk gradient ratio (~8× between omega=0.8 and floor=0.1) sharper
           than the original 0.3 (~2.7×).

  Fix 3 — _dist_to_best_unknown: proximity tie-breaking.
           When multiple unknown sectors share the maximum weight (equal omega
           at episode start), each UAV now targets the nearest one among those
           tied-max sectors rather than always sector_0.  This gives natural
           quadrant assignment without any hard-coded logic, and does not
           override the risk-potential field when sectors have distinct weights.

  Fix 4 — PBRS_SCALE: 1.0 → 2.0
           At PBRS_SCALE=1.0 the total navigation reward (~8–12 units/day/UAV)
           was dominated by SAFE_RETURN_BONUS(10.0) in the notebook.  After
           fixing SAFE_RETURN_BONUS to 1.0 (notebook fix), PBRS_SCALE=2.0
           ensures exploration rewards are still meaningfully larger than the
           safe-return bonus without exceeding INFECTED_FOUND_BONUS(30), which
           would create a "rush past sectors without hovering" incentive.
           (Hovering generates PBRS=0, not negative, so diagnosis is preserved.)
"""

import os
import numpy as np
import pandas as pd

# ─── GRID / EPISODE CONSTANTS ─────────────────────────────────────────────────

GRID_ROWS  = 10
GRID_COLS  = 10
N_SECTORS  = GRID_ROWS * GRID_COLS   # 100
N_UAVS     = 4
T_MAX      = 72                       # episode length in DAYS only

# ─── UAV PHYSICS ──────────────────────────────────────────────────────────────

E_MAX           = 40.0    # daily flight budget
E_MOVE          = 1.0     # energy per move step
E_HOVER         = 1.21    # energy per STAY step
TAU_DIAG        = 2       # consecutive STAY steps needed to diagnose a sector
DAILY_STEPS_MAX = 40      # inner-loop step limit per day

# ─── CRASH / RETURN CONSTANTS ────────────────────────────────────────────────

CRASH_PENALTY             = 100.0   # one-time penalty on crash (applied by training loop)
RETURN_BUFFER             = 3       # safety margin in steps for survival ratio
SURVIVAL_RATIO_THRESHOLD  = 1.2     # explore/return mode boundary

# ─── RISK / REWARD PARAMETERS ────────────────────────────────────────────────

GAMMA      = 0.8    # detection history decay (Eq. 2)
ETA        = 0.03   # urgency growth rate for healthy sectors (Eq. 5)
ALPHA      = 0.4    # history bias in Omega (Eq. 4)
SIGMA      = 2.0    # spatial diffusion radius
H_MAX      = 10.0   # detection history saturation

PSI             = 1.0    # risk coverage weight (Eq. 11)
LAMBDA_ENG      = 0.1    # energy penalty weight
ZETA            = 5.0    # repulsion penalty (CVT separation)
SIGMA_REP       = 2.0    # repulsion radius
EPSILON         = 1.0    # distance offset — keeps reward bounded
PBRS_SCALE      = 2.0    # Fix 4: PBRS coefficient (was 1.0)
W_UNKNOWN_FLOOR = 0.1    # Fix 2: minimum risk weight for unknowns (was 0.0)

# ─── TREATMENT ────────────────────────────────────────────────────────────────
TREATMENT_DAYS = 3

# ─── OBSERVATION SIZE ─────────────────────────────────────────────────────────
# own(9) + risk_weights(100) + status_norm(100) + other_uavs(6)
OBS_SIZE   = 9 + N_SECTORS + N_SECTORS + (N_UAVS - 1) * 2   # 215
JOINT_SIZE = N_UAVS * OBS_SIZE                               # 860


# ─── ENVIRONMENT ──────────────────────────────────────────────────────────────

class UAVFieldEnv:
    """
    10×10 multi-UAV crop disease monitoring environment with daily sorties.

    4 UAVs start at the four corners (their home Access Points):
      UAV 0 → (0, 0)          top-left
      UAV 1 → (0, 9)          top-right
      UAV 2 → (9, 0)          bottom-left
      UAV 3 → (9, 9)          bottom-right

    Each day has up to DAILY_STEPS_MAX=40 inner steps. At end-of-day,
    UAVs return to APs, energy recharges, and disease advances one day.
    """

    def __init__(self, sim_log_path, grid_config_path, dataset_dir=None):
        sim_log      = pd.read_csv(sim_log_path)
        self.T       = int(sim_log.time_step.max())

        sim_sorted         = sim_log.sort_values(['time_step', 'sector_id'])
        _base_lookup       = (sim_sorted['true_status']
                              .values.astype(np.int8)
                              .reshape(self.T + 1, N_SECTORS))

        # ── Dataset mode ─────────────────────────────────────────────────
        if dataset_dir is not None and dataset_dir.endswith('.npy') and os.path.isfile(dataset_dir):
            self._all_lookups = np.load(dataset_dir).astype(np.int8)
            print(f"[UAVFieldEnv] Dataset loaded: "
                  f"{self._all_lookups.shape} int8 array "
                  f"({self._all_lookups.nbytes // 1_000_000} MB)", flush=True)
        elif dataset_dir is not None and os.path.isdir(dataset_dir):
            csv_files = sorted(
                f for f in os.listdir(dataset_dir)
                if f.startswith('sim_') and f.endswith('.csv')
            )
            if csv_files:
                print(f"[UAVFieldEnv] Loading {len(csv_files)} simulations "
                      f"from {dataset_dir} ...", flush=True)
                sims = []
                for fname in csv_files:
                    df = pd.read_csv(os.path.join(dataset_dir, fname))
                    df_s = df.sort_values(['time_step', 'sector_id'])
                    arr  = (df_s['true_status']
                            .values.astype(np.int8)
                            .reshape(self.T + 1, N_SECTORS))
                    sims.append(arr)
                self._all_lookups = np.stack(sims, axis=0)
                print(f"[UAVFieldEnv] Dataset loaded: "
                      f"{self._all_lookups.shape} int8 array "
                      f"({self._all_lookups.nbytes // 1_000_000} MB)", flush=True)
            else:
                self._all_lookups = None
        else:
            self._all_lookups = None

        self._base_lookup  = _base_lookup
        self.status_lookup = _base_lookup

        self.sector_pos = {
            sid: (sid // GRID_COLS, sid % GRID_COLS)
            for sid in range(N_SECTORS)
        }
        self.pos_to_sid = {v: k for k, v in self.sector_pos.items()}
        self.neighbors  = self._build_neighbors()

        self.sector_rows = np.arange(N_SECTORS, dtype=np.float32) // GRID_COLS
        self.sector_cols = np.arange(N_SECTORS, dtype=np.float32) %  GRID_COLS
        self.sector_rc   = np.stack([self.sector_rows,
                                     self.sector_cols], axis=1)

        self._two_sigma2     = 2.0 * SIGMA     ** 2
        self._two_sigma_rep2 = 2.0 * SIGMA_REP ** 2

        # Fixed home Access Points for each UAV
        self.ap_pos = [
            (0.0, 0.0),
            (0.0, float(GRID_COLS - 1)),
            (float(GRID_ROWS - 1), 0.0),
            (float(GRID_ROWS - 1), float(GRID_COLS - 1)),
        ]

        self.reset()

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self):
        # 1. Resample trajectory if dataset is loaded
        if self._all_lookups is not None:
            idx = np.random.randint(len(self._all_lookups))
            self.status_lookup = self._all_lookups[idx]
        else:
            self.status_lookup = self._base_lookup

        # Day and intra-day counters
        self.current_day = 0
        self.daily_step  = 0

        self.t           = 0
        self.true_status = self._load_true_status(0)

        # 2. UAV knowledge and risk dynamics
        self.uav_status  = np.full(N_SECTORS, 2, dtype=int)
        self.H           = np.zeros(N_SECTORS, dtype=float)
        self.last_visit  = np.zeros(N_SECTORS, dtype=int)

        # 3. Continuous kinematics — start at APs
        self.uav_pos = [
            (0.0, 0.0),
            (0.0, float(GRID_COLS - 1)),
            (float(GRID_ROWS - 1), 0.0),
            (float(GRID_ROWS - 1), float(GRID_COLS - 1)),
        ]
        self.last_v = [np.zeros(2, dtype=np.float32) for _ in range(N_UAVS)]

        # 4. Resource and diagnostic counters
        self.energy = [float(E_MAX)] * N_UAVS
        self.dwell  = [0] * N_UAVS

        # Crashed flag for the current day
        self.crashed = [False] * N_UAVS
        # Tracks which UAVs crashed THIS step (for training loop CRASH_PENALTY)
        self.newly_crashed = [False] * N_UAVS
        # Set True at daily reset for UAVs that were at their AP
        self.safely_returned = [False] * N_UAVS

        # 5. Treatment and intervention tracking
        self.treatment_timer   = np.zeros(N_SECTORS, dtype=int)
        self.ever_diagnosed    = np.zeros(N_SECTORS, dtype=bool)
        self.ever_infected     = np.zeros(N_SECTORS, dtype=bool)
        self.intervention_mask = np.zeros(N_SECTORS, dtype=bool)

        # 6. Compute initial potential field weights
        self.w = self._compute_risk_weights()

        # PBRS distance trackers
        self.last_dist_to_target = [self._dist_to_best_unknown(u) for u in range(N_UAVS)]
        self.last_dist_to_ap     = [self._dist_to_ap(u) for u in range(N_UAVS)]

        return self._get_all_obs()

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self, actions):
        assert len(actions) == N_UAVS
        energy_consumed = [0.0] * N_UAVS

        # Clear per-step crash/return flags
        self.newly_crashed   = [False] * N_UAVS
        self.safely_returned = [False] * N_UAVS

        # ── Execute Continuous Actions ────────────────────────────────────────
        for u in range(N_UAVS):
            # Skip action execution for crashed UAVs
            if self.crashed[u]:
                continue

            v_u = np.array(actions[u], dtype=np.float32)
            v_u = np.clip(v_u, -1.0, 1.0)
            v_mag = np.linalg.norm(v_u)

            if v_mag > 1.0:
                v_u = v_u / v_mag
                v_mag = 1.0

            # Update position
            r, c = self.uav_pos[u]
            nr = np.clip(r + v_u[0], 0, GRID_ROWS - 1)
            nc = np.clip(c + v_u[1], 0, GRID_COLS - 1)
            self.uav_pos[u] = (float(nr), float(nc))

            # Energy and dwell
            if v_mag < 0.35:
                self.dwell[u]      += 1
                energy_consumed[u]  = E_HOVER
            else:
                self.dwell[u]       = 0
                energy_consumed[u]  = E_MOVE

            self.energy[u] = max(0.0, self.energy[u] - energy_consumed[u])
            self.last_v[u] = v_u

            # Crash check fires BEFORE diagnosis
            if self.energy[u] <= 0 and not self._at_ap(u):
                self.crashed[u]       = True
                self.newly_crashed[u] = True
                self.dwell[u]         = 0
                self.last_v[u]        = np.zeros(2, dtype=np.float32)

        # ── Diagnosis check ───────────────────────────────────────────────────
        for u in range(N_UAVS):
            # Only non-crashed UAVs can diagnose
            if self.crashed[u]:
                continue

            if self.dwell[u] >= TAU_DIAG:
                r_cont, c_cont = self.uav_pos[u]
                r_int = int(np.clip(round(r_cont), 0, GRID_ROWS - 1))
                c_int = int(np.clip(round(c_cont), 0, GRID_COLS - 1))
                sid = self.pos_to_sid[(r_int, c_int)]

                if self.uav_status[sid] == 2:
                    self.uav_status[sid] = int(self.true_status[sid])
                    self.ever_diagnosed[sid] = True

                    if self.uav_status[sid] == 1:
                        self.ever_infected[sid] = True
                        self.treatment_timer[sid] = TREATMENT_DAYS

                b_kt = 1 if (self.true_status[sid] == 1 and
                             self.uav_status[sid] == 1) else 0
                self.H[sid]          = min(H_MAX, GAMMA * self.H[sid] + b_kt)
                self.last_visit[sid] = self.current_day

        # ── Advance intra-day step ────────────────────────────────────────────
        self.daily_step += 1

        # ── Compute current PBRS distances before reward ──────────────────────
        curr_dist_to_target = [self._dist_to_best_unknown(u) for u in range(N_UAVS)]
        curr_dist_to_ap     = [self._dist_to_ap(u) for u in range(N_UAVS)]
        survival_ratios     = [self._survival_ratio(u) for u in range(N_UAVS)]

        # ── Compute PBRS rewards ──────────────────────────────────────────────
        self.w   = self._compute_risk_weights()
        rewards  = [self._compute_reward(u, energy_consumed[u],
                                         curr_dist_to_target[u],
                                         curr_dist_to_ap[u],
                                         survival_ratios[u])
                    for u in range(N_UAVS)]

        # Store PBRS distances for next step
        self.last_dist_to_target = curr_dist_to_target
        self.last_dist_to_ap     = curr_dist_to_ap

        # ── Daily reset block ─────────────────────────────────────────────────
        if self.daily_step >= DAILY_STEPS_MAX:
            self._daily_reset()

        # ── Done condition ────────────────────────────────────────────────────
        done = (self.current_day >= self.T)

        obs  = self._get_all_obs()
        info = {
            "t":                self.current_day * DAILY_STEPS_MAX + self.daily_step,
            "current_day":      self.current_day,
            "daily_step":       self.daily_step,
            "uav_pos":          list(self.uav_pos),
            "energy":           list(self.energy),
            "uav_status":       self.uav_status.copy(),
            "true_status":      self.true_status.copy(),
            "risk_weights":     self.w.copy(),
            "dwell":            list(self.dwell),
            "treatment_timer":  self.treatment_timer.copy(),
            "newly_crashed":    list(self.newly_crashed),
            "safely_returned":  list(self.safely_returned),
            "survival_ratio":   survival_ratios,
        }
        return obs, rewards, done, info

    # ── Daily Reset ───────────────────────────────────────────────────────────

    def _daily_reset(self):
        """End-of-day processing: crashes, disease advance, recharge, return."""

        # Time-out crash — UAVs not at AP at end of day
        for u in range(N_UAVS):
            if not self.crashed[u] and not self._at_ap(u):
                self.crashed[u]       = True
                self.newly_crashed[u] = True

        # Safe return tracking
        for u in range(N_UAVS):
            if self._at_ap(u) and not self.crashed[u]:
                self.safely_returned[u] = True

        # Disease advance
        self.current_day += 1

        if self.current_day <= self.T:
            prev_raw_status = self._load_true_status(self.current_day - 1)
            raw_status      = self._load_true_status(min(self.current_day, self.T))

            # Reset mask if a sector catches disease again naturally
            new_outbreaks = (prev_raw_status == 0) & (raw_status == 1)
            self.intervention_mask[new_outbreaks] = False

            self.true_status = raw_status.copy()
            self.true_status[self.intervention_mask] = 0

        # Treatment countdown — once per day
        active_tx = (self.uav_status == 1) & (self.treatment_timer > 0)
        self.treatment_timer[active_tx] -= 1

        newly_healed = (self.uav_status == 1) & (self.treatment_timer == 0)
        if newly_healed.any():
            self.uav_status[newly_healed]        = 0
            self.true_status[newly_healed]       = 0
            self.intervention_mask[newly_healed] = True

        # Re-infection detection — once per day
        re_infected = (self.uav_status == 0) & (self.true_status == 1)
        if re_infected.any():
            self.uav_status[re_infected]      = 2
            self.treatment_timer[re_infected] = 0

        # H decay — once per day
        visited_today = (self.last_visit == self.current_day - 1)
        unvisited     = ~visited_today
        self.H[unvisited] *= GAMMA

        # Recharge energy
        self.energy = [float(E_MAX)] * N_UAVS

        # Return all UAVs to APs
        for u in range(N_UAVS):
            self.uav_pos[u] = self.ap_pos[u]

        # Reset intra-day counters
        self.dwell   = [0] * N_UAVS
        self.crashed = [False] * N_UAVS
        self.last_v  = [np.zeros(2, dtype=np.float32) for _ in range(N_UAVS)]

        # Recompute risk weights
        self.w = self._compute_risk_weights()

        # Reset PBRS distances for the new day
        self.last_dist_to_target = [self._dist_to_best_unknown(u) for u in range(N_UAVS)]
        self.last_dist_to_ap     = [self._dist_to_ap(u) for u in range(N_UAVS)]

        # Reset daily step counter
        self.daily_step = 0

    # ── PBRS Reward ───────────────────────────────────────────────────────────

    def _compute_reward(self, u, energy_consumed,
                        curr_dist_to_target, curr_dist_to_ap, survival_ratio):
        """
        PBRS-based reward.

        Explore mode (survival_ratio > SURVIVAL_RATIO_THRESHOLD):
            r_nav = PBRS_SCALE × (last_dist_to_target − curr_dist_to_target)
        Return mode:
            r_nav = PBRS_SCALE × (last_dist_to_ap − curr_dist_to_ap)
        Always:
            − LAMBDA_ENG × energy_consumed − ZETA × repulsion

        Fix 1: CRASH_PENALTY is NOT applied here. It is applied exactly once
        by the training loop via info['newly_crashed'], covering both
        energy-depletion crashes (detected in step()) and time-out crashes
        (detected in _daily_reset()). Applying it here too was causing
        energy-depletion crashes to be penalised -200 instead of -100.
        """
        # PBRS navigation reward
        if survival_ratio > SURVIVAL_RATIO_THRESHOLD:
            # Explore mode: reward approach toward highest-risk unknown sector
            r_nav = PBRS_SCALE * (self.last_dist_to_target[u] - curr_dist_to_target)
        else:
            # Return mode: reward approach toward home AP
            r_nav = PBRS_SCALE * (self.last_dist_to_ap[u] - curr_dist_to_ap)

        repulsion = self._compute_repulsion(u)
        return r_nav - LAMBDA_ENG * energy_consumed - ZETA * repulsion

    # ── Helper: distance to best unknown ──────────────────────────────────────

    def _dist_to_best_unknown(self, u):
        """
        Distance from UAV u to the best unknown sector.

        Fix 3: Among sectors tied at the maximum weight (within tolerance),
        the nearest one to this UAV is chosen.  When all unknowns have equal
        weight (e.g., W_UNKNOWN_FLOOR=0.1 at episode start), each UAV
        naturally targets its closest unknown, spreading the four UAVs toward
        their respective quadrants instead of all converging on sector_0.
        When weights are distinct, argmax still selects the globally highest-
        risk sector as before.
        """
        unknown_mask = (self.uav_status == 2)
        if not unknown_mask.any():
            return 0.0

        unknown_idx = np.where(unknown_mask)[0]
        w_unknown   = self.w[unknown_idx]
        w_max       = w_unknown.max()

        # Find all sectors at (or within float tolerance of) the maximum weight
        top_mask = w_unknown >= (w_max - 1e-6)
        top_idx  = unknown_idx[top_mask]

        # Among tied-max sectors, pick the nearest to this UAV
        r_u, c_u = self.uav_pos[u]
        dists    = np.sqrt((self.sector_rows[top_idx] - r_u) ** 2
                          + (self.sector_cols[top_idx] - c_u) ** 2)
        nearest  = top_idx[np.argmin(dists)]

        r_n, c_n = self.sector_pos[nearest]
        return np.sqrt((r_u - r_n) ** 2 + (c_u - c_n) ** 2)

    # ── Helper: survival ratio ────────────────────────────────────────────────

    def _survival_ratio(self, u):
        """energy[u] / max(dist_to_AP * E_MOVE + RETURN_BUFFER, 1e-6)"""
        dist = self._dist_to_ap(u)
        return self.energy[u] / max(dist * E_MOVE + RETURN_BUFFER, 1e-6)

    # ── Helper: distance to AP ────────────────────────────────────────────────

    def _dist_to_ap(self, u):
        """Euclidean distance from UAV u to its home AP."""
        r_u, c_u = self.uav_pos[u]
        r_a, c_a = self.ap_pos[u]
        return np.sqrt((r_u - r_a) ** 2 + (c_u - c_a) ** 2)

    # ── Helper: at AP check ───────────────────────────────────────────────────

    def _at_ap(self, u):
        """True if UAV u is within 0.5 grid units of its home AP."""
        return self._dist_to_ap(u) < 0.5

    # ── Risk Computations ─────────────────────────────────────────────────────

    def _compute_risk_weights(self):
        """Eq. 5 — dynamic risk weight per sector (fully vectorized)."""
        w            = np.zeros(N_SECTORS, dtype=np.float32)
        infected_m   = (self.uav_status == 1)
        healthy_m    = (self.uav_status == 0)
        unknown_m    = (self.uav_status == 2)

        # Infected → always 1.0
        w[infected_m] = 1.0

        # Healthy → min(1, η × Δt)
        if healthy_m.any():
            delta_t      = self.current_day - self.last_visit[healthy_m]
            w[healthy_m] = np.minimum(1.0, ETA * delta_t)

        # Unknown → max(W_UNKNOWN_FLOOR, min(1, Omega))
        # Fix 2: W_UNKNOWN_FLOOR=0.1 prevents degenerate all-zeros argmax
        # while keeping the risk gradient ratio (~8×) sharper than 0.3 (~2.7×).
        if unknown_m.any():
            omega        = self._compute_omega_batch(unknown_m, infected_m)
            w[unknown_m] = np.maximum(W_UNKNOWN_FLOOR,
                                      np.minimum(1.0, omega))
        return w

    def _compute_omega_batch(self, unknown_mask, infected_mask):
        """Eq. 4 — computes Omega for ALL unknown sectors simultaneously."""
        unk_idx = np.where(unknown_mask)[0]
        history = ALPHA * (self.H[unk_idx] / H_MAX)

        inf_idx = np.where(infected_mask)[0]
        if inf_idx.size == 0:
            return history

        diff    = (self.sector_rc[unk_idx, np.newaxis, :]
                   - self.sector_rc[np.newaxis, inf_idx, :])
        dist_sq = (diff ** 2).sum(axis=2)
        spatial = np.exp(-dist_sq / self._two_sigma2).sum(axis=1)

        return history + (1 - ALPHA) * spatial

    # ── Repulsion ─────────────────────────────────────────────────────────────

    def _compute_repulsion(self, u):
        """Eq. 9 — inter-UAV repulsion penalty (vectorized over other UAVs)."""
        r_u, c_u  = self.uav_pos[u]
        others    = [j for j in range(N_UAVS) if j != u]
        r_others  = np.array([self.uav_pos[j][0] for j in others], dtype=np.float32)
        c_others  = np.array([self.uav_pos[j][1] for j in others], dtype=np.float32)
        dist_sq   = (r_others - r_u) ** 2 + (c_others - c_u) ** 2
        return float(np.sum(np.exp(-dist_sq / self._two_sigma_rep2)))

    # ── Observations ──────────────────────────────────────────────────────────

    def _get_obs(self, u):
        """
        Returns flat observation vector for UAV u (OBS_SIZE = 215 values).

          [0:9]      own: row_norm, col_norm, energy_norm, vx, vy,
                          dist_to_AP_row_norm, dist_to_AP_col_norm,
                          survival_ratio_clipped, daily_step_norm
          [9:109]    risk_weight per sector
          [109:209]  uav_status normalised (0=H, 0.5=I, 1.0=?)
          [209:215]  Δrow/Δcol for each other UAV (3 × 2)
        """
        r_u, c_u = self.uav_pos[u]
        r_a, c_a = self.ap_pos[u]

        sr = self._survival_ratio(u)

        own = np.array([
            r_u / (GRID_ROWS - 1),
            c_u / (GRID_COLS - 1),
            self.energy[u] / E_MAX,
            self.last_v[u][0],
            self.last_v[u][1],
            (r_u - r_a) / (GRID_ROWS - 1),
            (c_u - c_a) / (GRID_COLS - 1),
            np.clip(sr, 0, 5),
            self.daily_step / DAILY_STEPS_MAX,
        ], dtype=np.float32)

        risk        = self.w.astype(np.float32)
        status_norm = (self.uav_status / 2.0).astype(np.float32)

        other_pos = []
        for j in range(N_UAVS):
            if j == u:
                continue
            r_j, c_j = self.uav_pos[j]
            other_pos.append((r_j - r_u) / (GRID_ROWS - 1))
            other_pos.append((c_j - c_u) / (GRID_COLS - 1))
        other_pos = np.array(other_pos, dtype=np.float32)

        return np.concatenate([own, risk, status_norm, other_pos])

    def _get_all_obs(self):
        return [self._get_obs(u) for u in range(N_UAVS)]

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _load_true_status(self, t):
        """O(1) lookup into pre-indexed status array."""
        t_clamped = min(t, self.T)
        return self.status_lookup[t_clamped].copy()

    def _build_neighbors(self):
        neighbors = {}
        for sid in range(N_SECTORS):
            r, c  = self.sector_pos[sid]
            nbrs  = []
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < GRID_ROWS and 0 <= nc < GRID_COLS:
                        nbrs.append(self.pos_to_sid[(nr, nc)])
            neighbors[sid] = nbrs
        return neighbors

    def get_grid_summary(self):
        """Returns a compact ASCII summary of the current grid state."""
        pos_set = {}
        for u in range(N_UAVS):
            r_int = int(round(self.uav_pos[u][0]))
            c_int = int(round(self.uav_pos[u][1]))
            pos_set[(r_int, c_int)] = u

        lines = [f"\nday={self.current_day} step={self.daily_step}  "
                 + "  ".join(f"UAV{u}@({self.uav_pos[u][0]:.1f},{self.uav_pos[u][1]:.1f})"
                             f" E={self.energy[u]:.0f}"
                             f"{'[X]' if self.crashed[u] else ''}"
                             for u in range(N_UAVS))]
        header = "     " + "".join(f"{c:3}" for c in range(GRID_COLS))
        lines.append(header)
        for r in range(GRID_ROWS):
            row_str = f"r{r:2}  "
            for c in range(GRID_COLS):
                sid = self.pos_to_sid[(r, c)]
                sym = ["H", "I", "?"][self.uav_status[sid]]
                if (r, c) in pos_set:
                    sym = str(pos_set[(r, c)])
                row_str += f"{sym:>3}"
            lines.append(row_str)
        return "\n".join(lines)

    @property
    def total_steps(self):
        """Total inner steps in an episode: T days × DAILY_STEPS_MAX steps/day."""
        return self.T * DAILY_STEPS_MAX

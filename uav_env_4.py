"""
uav_env_4.py  (dynamic environment — Eq. 11 reward, runtime disease spread)

Gym-style environment for the Multi-UAV Active Sensing system.
10x10 grid, 4 UAVs, 72-day episodes with intra-day sortie loops.

Key changes from uav_env_3.py
──────────────────────────────
  1. Dynamic disease spread — computed at runtime each day using wind,
     humidity, and seasonality (ported from simulate_disease.py /
     generate_dataset.py).  No pre-baked lookup tables.

  2. Eq. 11 reward — used as an exploration potential field.
     Step reward uses the potential difference, preventing state-reward
     farming while preserving the paper's continuous risk field.

  3. Unified reward economy — discovery bonuses, crash penalties, safe-return
     bonuses, and overhover penalties all live inside the environment with
     balanced magnitudes (~8x per-step reward for discoveries).

  4. Environmental observations — wind_speed, wind_dir (sin/cos), humidity,
     season added to the observation vector (OBS_SIZE 215 -> 220).

  5. Per-episode diversity — random initial infection seeds (1-3 non-corner
     sectors) and random wind base direction each reset().

Observation vector per UAV (220 values):
  [row_norm, col_norm, energy_norm, vx, vy,          (5)  own state
   dist_to_AP_row_norm, dist_to_AP_col_norm,         (2)  AP-relative
   survival_ratio_clipped, daily_step_norm,           (2)  sortie phase
   wind_speed_norm, wind_dir_sin, wind_dir_cos,      (3)  environment
   humidity_norm, season_norm,                        (2)  environment
   risk_w[0..99],                                   (100) risk weights
   status_norm[0..99],                              (100) 0=H, 0.5=I, 1.0=?
   delta_row_j/delta_col_j for each other UAV]        (6) 3 others x 2
"""

import math
import numpy as np

# ─── GRID / EPISODE CONSTANTS ─────────────────────────────────────────────────

GRID_ROWS  = 10
GRID_COLS  = 10
N_SECTORS  = GRID_ROWS * GRID_COLS   # 100
N_UAVS    = 4
T_MAX      = 72                       # episode length in days

# ─── UAV PHYSICS ──────────────────────────────────────────────────────────────

E_MAX           = 150.0    # daily flight budget
E_MOVE          = 1.0      # energy per move step
E_HOVER         = 1.21     # energy per STAY step
TAU_DIAG        = 2        # consecutive STAY steps needed to diagnose
DAILY_STEPS_MAX = 80       # inner-loop step limit per day

# ─── CRASH / RETURN CONSTANTS ────────────────────────────────────────────────

RETURN_BUFFER             = 5       # extra steps of margin when computing return budget
SURVIVAL_RATIO_THRESHOLD  = 1.5     # explore/return mode boundary

# ─── DISEASE SPREAD (from simulate_disease.py / generate_dataset.py) ─────────

HEALING_PERIOD = 20       # natural self-healing days (without UAV treatment)
SPREAD_ALPHA   = 0.015    # base contact spread weight
SPREAD_BETA    = 0.03     # wind-driven spread weight

# Corner sectors excluded from initial infection seeding (UAV start positions)
_CORNER_SIDS = {0, GRID_COLS - 1, N_SECTORS - GRID_COLS, N_SECTORS - 1}

# ─── RISK / REWARD PARAMETERS ────────────────────────────────────────────────

GAMMA      = 0.8    # detection history decay (Eq. 2)
ETA        = 0.03   # urgency growth rate for healthy sectors (Eq. 5)
ALPHA      = 0.4    # history bias in Omega (Eq. 4)
SIGMA      = 2.0    # spatial diffusion radius
H_MAX      = 10.0   # detection history saturation

PSI             = 1.0    # risk coverage weight (Eq. 11)
LAMBDA_ENG      = 0.1    # energy penalty weight
ZETA            = 1.0    # repulsion penalty (CVT separation)
SIGMA_REP       = 2.0    # repulsion radius
EPSILON         = 1.0    # distance offset — keeps reward bounded
W_UNKNOWN_FLOOR = 0.1    # minimum risk weight for unknowns

# ─── TREATMENT ────────────────────────────────────────────────────────────────

TREATMENT_DAYS = 3

# ─── UNIFIED REWARD ECONOMY (moved from notebook, rebalanced) ────────────────

INFECTED_FOUND_BONUS    = 40.0    # ~8x per-step reward (~5)
NEW_OUTBREAK_MULTIPLIER = 2.0     # 80.0 for genuinely new outbreak
HEALTHY_FOUND_BONUS     = 5.0     # first diagnosis of healthy sector
CRASH_PENALTY           = 50.0    # one-time on crash (raised — crashes should now be rare)
SAFE_RETURN_BONUS       = 2.0     # small bonus for returning safely
OVERHOVER_PENALTY       = 0.5     # per step hovering on already-diagnosed sector
DIAGNOSED_INFECTED_DECAY = 0.1    # diagnosed hotspots contribute less to explore potential
RETURN_POTENTIAL_SCALE   = 6.0    # dense AP-return gradient when survival is low

# ─── OBSERVATION SIZE ─────────────────────────────────────────────────────────

ENV_OBS_DIM = 5   # wind_speed_norm, wind_dir_sin, wind_dir_cos, humidity_norm, season_norm
OBS_SIZE    = 9 + ENV_OBS_DIM + N_SECTORS + N_SECTORS + (N_UAVS - 1) * 2   # 220
JOINT_SIZE  = N_UAVS * OBS_SIZE                                             # 880


# ─── ENVIRONMENT ──────────────────────────────────────────────────────────────

class UAVFieldEnv:
    """
    10x10 multi-UAV crop disease monitoring environment with daily sorties
    and dynamic disease spread.

    4 UAVs start at the four corners (their home Access Points):
      UAV 0 -> (0, 0)          top-left
      UAV 1 -> (0, 9)          top-right
      UAV 2 -> (9, 0)          bottom-left
      UAV 3 -> (9, 9)          bottom-right

    Each day has up to DAILY_STEPS_MAX inner steps. At end-of-day,
    UAVs return to APs, energy recharges, and disease advances one day
    using the runtime spread model (wind, humidity, seasonality).
    """

    def __init__(self, seed=None):
        self.T = T_MAX

        # Sector geometry (same as uav_env_3)
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

        # Non-corner sector IDs for initial infection seeding
        self._candidate_sids = np.array(
            [sid for sid in range(N_SECTORS) if sid not in _CORNER_SIDS],
            dtype=np.int32
        )

        self.rng = np.random.default_rng(seed)
        self.last_reward_components = [{} for _ in range(N_UAVS)]
        self.reset()

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self):
        # Day and intra-day counters
        self.current_day = 0
        self.daily_step  = 0

        # Random initial infection: 1-3 non-corner sectors
        n_seeds = int(self.rng.integers(1, 4))
        init_seeds = self.rng.choice(
            self._candidate_sids, size=n_seeds, replace=False
        )

        # True disease status (0=healthy, 1=infected)
        self.true_status = np.zeros(N_SECTORS, dtype=np.int8)
        self.true_status[init_seeds] = 1

        # Natural healing timer (separate from UAV treatment timer)
        self.healing_timer = np.zeros(N_SECTORS, dtype=np.int16)
        self.healing_timer[init_seeds] = HEALING_PERIOD

        # Per-episode wind diversity
        self.wind_base_dir = float(self.rng.uniform(0, 360))

        # Generate day-0 environmental variables
        self._generate_env_vars(0)

        # UAV knowledge and risk dynamics
        self.uav_status  = np.full(N_SECTORS, 2, dtype=int)
        self.H           = np.zeros(N_SECTORS, dtype=float)
        self.last_visit  = np.zeros(N_SECTORS, dtype=int)

        # Continuous kinematics — start at APs
        self.uav_pos = [
            (0.0, 0.0),
            (0.0, float(GRID_COLS - 1)),
            (float(GRID_ROWS - 1), 0.0),
            (float(GRID_ROWS - 1), float(GRID_COLS - 1)),
        ]
        self.last_v = [np.zeros(2, dtype=np.float32) for _ in range(N_UAVS)]

        # Resource and diagnostic counters
        self.energy = [float(E_MAX)] * N_UAVS
        self.dwell  = [0] * N_UAVS

        # Crashed flag for the current day
        self.crashed = [False] * N_UAVS
        self.newly_crashed    = [False] * N_UAVS
        self.safely_returned  = [False] * N_UAVS
        self.left_ap_today    = [False] * N_UAVS

        # Treatment and intervention tracking
        self.treatment_timer   = np.zeros(N_SECTORS, dtype=int)
        self.ever_diagnosed    = np.zeros(N_SECTORS, dtype=bool)
        self.ever_infected     = np.zeros(N_SECTORS, dtype=bool)
        self.intervention_mask = np.zeros(N_SECTORS, dtype=bool)

        # Diagnosis attribution for reward (cleared each step)
        self._newly_diagnosed_by = {}
        self._ever_infected_before_step  = np.zeros(N_SECTORS, dtype=bool)
        self._ever_diagnosed_before_step = np.zeros(N_SECTORS, dtype=bool)

        # Compute initial potential field weights
        self.w = self._compute_risk_weights()
        self.last_phi_explore = [self._phi_explore(u) for u in range(N_UAVS)]
        self.last_phi_return = [self._phi_return(u) for u in range(N_UAVS)]

        return self._get_all_obs()

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self, actions):
        assert len(actions) == N_UAVS
        energy_consumed = [0.0] * N_UAVS

        # Clear per-step flags
        self.newly_crashed   = [False] * N_UAVS
        self.safely_returned = [False] * N_UAVS
        self._newly_diagnosed_by = {}

        # Save pre-step state for discovery bonus attribution
        self._ever_infected_before_step  = self.ever_infected.copy()
        self._ever_diagnosed_before_step = self.ever_diagnosed.copy()

        # ── Execute Continuous Actions ────────────────────────────────────────
        for u in range(N_UAVS):
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
            if not self._at_ap(u):
                self.left_ap_today[u] = True

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

                    # Track which UAV diagnosed which sector
                    if u not in self._newly_diagnosed_by:
                        self._newly_diagnosed_by[u] = []
                    self._newly_diagnosed_by[u].append(sid)

                b_kt = 1 if (self.true_status[sid] == 1 and
                             self.uav_status[sid] == 1) else 0
                self.H[sid]          = min(H_MAX, GAMMA * self.H[sid] + b_kt)
                self.last_visit[sid] = self.current_day

        # ── Advance intra-day step ────────────────────────────────────────────
        self.daily_step += 1

        end_of_day = (self.daily_step >= DAILY_STEPS_MAX)
        if end_of_day:
            self._mark_end_of_day_outcomes()

        # ── Compute rewards (Eq. 11 + bonuses/penalties) ──────────────────────
        self.w = self._compute_risk_weights()
        rewards = [self._compute_reward(u, energy_consumed[u]) for u in range(N_UAVS)]

        # ── Daily reset block ─────────────────────────────────────────────────
        if end_of_day:
            self._daily_reset()

        # ── Done condition ────────────────────────────────────────────────────
        done = (self.current_day >= self.T)

        obs  = self._get_all_obs()
        survival_ratios = [self._survival_ratio(u) for u in range(N_UAVS)]
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
            "wind_speed":       self.wind_speed,
            "wind_dir":         self.wind_dir,
            "humidity":         self.humidity,
            "season_mult":      self.season_mult,
            "reward_components": [dict(x) for x in self.last_reward_components],
        }
        return obs, rewards, done, info

    # ── Daily Reset ───────────────────────────────────────────────────────────

    def _mark_end_of_day_outcomes(self):
        """Set same-step crash / safe-return flags before reward calculation."""
        # Time-out crash — UAVs not at AP at end of day
        for u in range(N_UAVS):
            if not self.crashed[u] and not self._at_ap(u):
                self.crashed[u]       = True
                self.newly_crashed[u] = True

        # Safe return tracking — only if UAV actually left the AP during the day
        for u in range(N_UAVS):
            if self._at_ap(u) and not self.crashed[u] and self.left_ap_today[u]:
                self.safely_returned[u] = True

    def _daily_reset(self):
        """End-of-day processing: disease advance, recharge, and return."""

        # Advance day counter
        self.current_day += 1

        if self.current_day <= self.T:
            # Expire post-treatment immunity for sectors no longer under active treatment
            self.intervention_mask[self.treatment_timer == 0] = False

            # Generate new environmental variables for this day
            self._generate_env_vars(self.current_day)

            # Dynamic disease spread
            self._advance_disease()

        # Treatment countdown — once per day (UAV-triggered treatment)
        active_tx = (self.uav_status == 1) & (self.treatment_timer > 0)
        self.treatment_timer[active_tx] -= 1

        newly_healed = (self.uav_status == 1) & (self.treatment_timer == 0) & active_tx
        if newly_healed.any():
            self.uav_status[newly_healed]        = 0
            self.true_status[newly_healed]       = 0
            self.intervention_mask[newly_healed] = True

        # Re-infection detection — once per day
        re_infected = (self.uav_status == 0) & (self.true_status == 1)
        if re_infected.any():
            self.uav_status[re_infected]      = 2
            self.treatment_timer[re_infected] = 0
            self.intervention_mask[re_infected] = False

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
        self.dwell         = [0] * N_UAVS
        self.crashed       = [False] * N_UAVS
        self.last_v        = [np.zeros(2, dtype=np.float32) for _ in range(N_UAVS)]
        self.left_ap_today = [False] * N_UAVS

        # Recompute risk weights
        self.w = self._compute_risk_weights()
        self.last_phi_explore = [self._phi_explore(u) for u in range(N_UAVS)]
        self.last_phi_return = [self._phi_return(u) for u in range(N_UAVS)]

        # Reset daily step counter
        self.daily_step = 0

    # ── Environmental Variables ───────────────────────────────────────────────

    def _generate_env_vars(self, day):
        """
        Generate daily environmental variables (ported from generate_dataset.py).
        Uses per-episode wind_base_dir for diversity.
        """
        self.wind_speed = float(np.clip(
            5.0 + self.rng.normal(0, 1.5), 1.0, 12.0
        ))

        base_dir = self.wind_base_dir + (90.0 / self.T) * day
        self.wind_dir = float(
            (base_dir + self.rng.normal(0, 10)) % 360
        )

        self.humidity = float(np.clip(
            60 + 25 * math.sin(math.pi * day / self.T)
            + self.rng.normal(0, 3),
            40, 100
        ))

        self.season_mult = float(np.clip(
            0.8 + 0.5 * math.sin(math.pi * day / self.T),
            0.5, 1.5
        ))

    # ── Disease Spread ────────────────────────────────────────────────────────

    @staticmethod
    def _contact_weight(row_j, col_j, row_k, col_k):
        """1.0 for edge-sharing neighbours, 0.5 for diagonal neighbours."""
        dr = abs(row_k - row_j)
        dc = abs(col_k - col_j)
        return 1.0 if (dr + dc == 1) else 0.5

    @staticmethod
    def _wind_alignment(row_j, col_j, row_k, col_k, wind_dir_deg):
        """
        Fraction of wind blowing FROM j TOWARD k.
        wind_alignment = max(0, cos(theta_wind - theta_jk))
        """
        dx       = col_k - col_j
        dy       = row_j - row_k   # north = negative row
        theta_jk = math.degrees(math.atan2(dy, dx))
        diff     = math.radians(wind_dir_deg - theta_jk)
        return max(0.0, math.cos(diff))

    def _compute_spread_prob(self, sid_k, infected_neighbors):
        """
        P(sector k gets infected this day) = 1 - prod(1 - p_j)
        over infected neighbours.
        """
        row_k, col_k = self.sector_pos[sid_k]
        survival = 1.0

        for j in infected_neighbors:
            row_j, col_j = self.sector_pos[j]
            contact_w = self._contact_weight(row_j, col_j, row_k, col_k)
            wind_w    = self._wind_alignment(row_j, col_j, row_k, col_k,
                                             self.wind_dir)
            p_j = min(
                (SPREAD_ALPHA * contact_w + SPREAD_BETA * wind_w)
                * (self.humidity / 100.0) * self.season_mult,
                0.95
            )
            survival *= (1.0 - p_j)

        return 1.0 - survival

    def _advance_disease(self):
        """
        Advance disease one day.  Called once per day in _daily_reset().

        Simultaneous update (snapshot-based, no chain reactions):
        - Healthy sectors with infected neighbours: roll spread probability
        - Infected sectors NOT under UAV treatment: decrement healing timer,
          heal if timer reaches 0
        - Intervention_mask sectors: immune to new infection (recently treated)
        """
        new_status = self.true_status.copy()
        new_timer  = self.healing_timer.copy()

        for sid in range(N_SECTORS):
            if self.true_status[sid] == 0:
                # Skip sectors with active intervention immunity
                if self.intervention_mask[sid]:
                    continue

                # Check for infected neighbours
                inf_nbrs = [n for n in self.neighbors[sid]
                            if self.true_status[n] == 1]
                if inf_nbrs:
                    p = self._compute_spread_prob(sid, inf_nbrs)
                    if self.rng.random() < p:
                        new_status[sid] = 1
                        new_timer[sid]  = HEALING_PERIOD

            elif self.true_status[sid] == 1:
                # Natural healing only for sectors NOT under UAV treatment
                if self.treatment_timer[sid] > 0:
                    continue  # UAV treatment handles this sector
                new_timer[sid] -= 1
                if new_timer[sid] <= 0:
                    new_status[sid] = 0
                    new_timer[sid]  = 0

        self.true_status   = new_status
        self.healing_timer = new_timer

    # ── Eq. 11 Reward ─────────────────────────────────────────────────────────

    def _compute_reward(self, u, energy_consumed):
        """
        Navigation reward = potential difference.

        Explore mode:
            phi_explore = PSI * sum_k effective_w_k / (dist_k + EPSILON)
            r_nav = phi_explore(t) - phi_explore(t-1)

        Return mode:
            phi_return = -RETURN_POTENTIAL_SCALE * dist_to_ap
            r_nav = phi_return(t) - phi_return(t-1)
        """
        if self.crashed[u] and not self.newly_crashed[u]:
            # Already crashed on a previous step — no reward
            return 0.0

        survival_ratio = self._survival_ratio(u)
        phi_explore_now = self._phi_explore(u)
        phi_return_now  = self._phi_return(u)

        if survival_ratio > SURVIVAL_RATIO_THRESHOLD:
            r_nav = phi_explore_now - self.last_phi_explore[u]
            nav_mode = "explore"
        else:
            r_nav = phi_return_now - self.last_phi_return[u]
            nav_mode = "return"

        # Energy penalty
        energy_penalty = LAMBDA_ENG * energy_consumed

        # Repulsion penalty (Eq. 9)
        repulsion = ZETA * self._compute_repulsion(u)

        reward = r_nav - energy_penalty - repulsion

        # Discovery bonus
        discovery_bonus = self._discovery_bonus(u)
        reward += discovery_bonus

        # Overhover penalty
        overhover_penalty = self._overhover_penalty(u)
        reward += overhover_penalty

        # Crash penalty (applied once on the step the UAV crashes)
        crash_penalty = 0.0
        if self.newly_crashed[u]:
            crash_penalty = CRASH_PENALTY
            reward -= crash_penalty

        # Safe return bonus
        safe_return_bonus = 0.0
        if self.safely_returned[u]:
            safe_return_bonus = SAFE_RETURN_BONUS
            reward += safe_return_bonus

        self.last_reward_components[u] = {
            "nav_mode": nav_mode,
            "nav_reward": float(r_nav),
            "phi_explore": float(phi_explore_now),
            "phi_return": float(phi_return_now),
            "survival_ratio": float(survival_ratio),
            "energy_penalty": float(energy_penalty),
            "repulsion": float(repulsion),
            "discovery_bonus": float(discovery_bonus),
            "overhover_penalty": float(overhover_penalty),
            "crash_penalty": float(crash_penalty),
            "safe_return_bonus": float(safe_return_bonus),
            "total_reward": float(reward),
        }
        self.last_phi_explore[u] = phi_explore_now
        self.last_phi_return[u] = phi_return_now

        return reward

    def _phi_explore(self, u):
        """Eq. 11-inspired explore potential, downweighting diagnosed hotspots."""
        r_u, c_u = self.uav_pos[u]
        dists = np.sqrt((self.sector_rows - r_u) ** 2 + (self.sector_cols - c_u) ** 2)

        effective_w = self.w.astype(np.float32).copy()
        diagnosed_infected = (self.uav_status == 1) & self.ever_diagnosed
        effective_w[diagnosed_infected] *= DIAGNOSED_INFECTED_DECAY

        return float(PSI * np.sum(effective_w / (dists + EPSILON)))

    def _phi_return(self, u):
        """Dense AP-return potential for low-survival mode."""
        return float(-RETURN_POTENTIAL_SCALE * self._dist_to_ap(u))

    def _discovery_bonus(self, u):
        """Compute discovery bonus for sectors UAV u diagnosed this step."""
        bonus = 0.0
        for sid in self._newly_diagnosed_by.get(u, []):
            if self.uav_status[sid] == 1:  # infected
                if not self._ever_infected_before_step[sid]:
                    bonus += INFECTED_FOUND_BONUS * NEW_OUTBREAK_MULTIPLIER
                else:
                    bonus += INFECTED_FOUND_BONUS
            elif self.uav_status[sid] == 0:  # healthy
                if not self._ever_diagnosed_before_step[sid]:
                    bonus += HEALTHY_FOUND_BONUS
        return bonus

    def _overhover_penalty(self, u):
        """Penalty for hovering on an already-diagnosed sector."""
        if self.crashed[u]:
            return 0.0
        r_cont, c_cont = self.uav_pos[u]
        r_int = int(np.clip(round(r_cont), 0, GRID_ROWS - 1))
        c_int = int(np.clip(round(c_cont), 0, GRID_COLS - 1))
        sid = self.pos_to_sid[(r_int, c_int)]
        if self.dwell[u] > TAU_DIAG and self.uav_status[sid] != 2:
            return -OVERHOVER_PENALTY
        return 0.0

    # ── Helper: survival ratio ────────────────────────────────────────────────

    def _survival_ratio(self, u):
        """
        steps_remaining / steps_needed_to_return.

        UAV max speed = 1 grid unit/step, so dist grid units ≈ dist steps.
        Adding RETURN_BUFFER gives a safety margin.

        > SURVIVAL_RATIO_THRESHOLD  →  explore mode
        < SURVIVAL_RATIO_THRESHOLD  →  return mode

        Unlike the old energy-based formula, this correctly triggers return
        mode as the day deadline approaches — regardless of energy level.
        """
        dist         = self._dist_to_ap(u)
        steps_needed = dist + RETURN_BUFFER
        steps_left   = DAILY_STEPS_MAX - self.daily_step
        return steps_left / max(steps_needed, 1e-6)

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

        # Infected -> always 1.0
        w[infected_m] = 1.0

        # Healthy -> min(1, eta * delta_t)
        if healthy_m.any():
            delta_t      = self.current_day - self.last_visit[healthy_m]
            w[healthy_m] = np.minimum(1.0, ETA * delta_t)

        # Unknown -> max(W_UNKNOWN_FLOOR, min(1, Omega))
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
        Returns flat observation vector for UAV u (OBS_SIZE = 220 values).

          [0:9]      own: row_norm, col_norm, energy_norm, vx, vy,
                          dist_to_AP_row_norm, dist_to_AP_col_norm,
                          survival_ratio_clipped, daily_step_norm
          [9:14]     env: wind_speed_norm, wind_dir_sin, wind_dir_cos,
                          humidity_norm, season_norm
          [14:114]   risk_weight per sector
          [114:214]  uav_status normalised (0=H, 0.5=I, 1.0=?)
          [214:220]  delta_row/delta_col for each other UAV (3 x 2)
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

        env_vars = np.array([
            self.wind_speed / 12.0,
            np.sin(np.radians(self.wind_dir)),
            np.cos(np.radians(self.wind_dir)),
            self.humidity / 100.0,
            (self.season_mult - 0.5) / 1.0,
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

        return np.concatenate([own, env_vars, risk, status_norm, other_pos])

    def _get_all_obs(self):
        return [self._get_obs(u) for u in range(N_UAVS)]

    # ── Helpers ───────────────────────────────────────────────────────────────

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
        lines.append(f"  wind={self.wind_speed:.1f}m/s dir={self.wind_dir:.0f}deg "
                     f"humidity={self.humidity:.0f}% season={self.season_mult:.2f}")
        return "\n".join(lines)

    @property
    def total_steps(self):
        """Total inner steps in an episode: T days x DAILY_STEPS_MAX steps/day."""
        return self.T * DAILY_STEPS_MAX

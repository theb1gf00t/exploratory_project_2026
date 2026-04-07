"""
generate_dataset.py  (scaled)

Generates 1,000 diverse Rice Blast disease-spread simulations for the
10×10 grid, each with a different random seed and initial infection
layout.  Saves the full dataset as a single .npy file:

    simulation_scaled/dataset.npy   shape (1000, 73, 100) int8

Disease control model
─────────────────────
  This dataset captures "background" disease dynamics — disease spreads
  between neighbouring sectors and self-heals after HEALING_PERIOD=7 days
  without any UAV intervention.

  In the training environment (uav_env.py), UAVs can diagnose infected
  sectors and trigger active treatment.  UAV-triggered treatment heals in
  TREATMENT_DAYS=3 days — ~4.7× faster than natural recovery.  The gap
  between these two numbers is the core incentive for sending UAVs.

Diversity levers
────────────────
  seeds_per_sim : 1–3 randomly chosen initial infection seeds
  seed_sectors  : chosen from all 100 sectors (corners avoided — UAVs
                  start there and an infected corner trivially inflates
                  early reward)
  rng_seed      : numpy seed = sim_index  (reproducible but diverse)
  wind_base_dir : each sim gets a random base wind direction (0–359°)
                  so spread direction varies across simulations

Usage:
    python generate_dataset.py
    python generate_dataset.py --n-sims 1000 --out-file path/to/dataset.npy
"""

import os
import sys
import math
import argparse
import numpy as np

# ─── CONSTANTS (must match simulate_disease.py) ───────────────────────────────

T              = 72
N_ROWS         = 10
N_COLS         = 10
N_SECTORS      = N_ROWS * N_COLS
HEALING_PERIOD = 20     # natural self-healing without UAV treatment.
                       # UAVs treat in TREATMENT_DAYS=3 days (uav_env.py),
                       # so active monitoring heals ~2.3× faster.
ALPHA          = 0.015  # base contact spread weight  (tuned for ~25 infected/day)
BETA           = 0.03  # wind-driven spread weight

# Corners (UAV start positions) — excluded from initial infection seeding
_CORNER_SIDS   = {0, N_COLS - 1, N_SECTORS - N_COLS, N_SECTORS - 1}

# ─── PATHS ────────────────────────────────────────────────────────────────────

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
ROOT_DIR    = os.path.dirname(PROJECT_DIR)
GRID_DIR    = os.path.join(ROOT_DIR, "grid_scaled")
JSON_PATH   = os.path.join(GRID_DIR, "grid_config.json")


# ─── ENV GENERATOR ────────────────────────────────────────────────────────────

def generate_env(t, T, wind_base_dir, rng):
    """Per-day environmental variables (same model as simulate_disease.py,
    but wind_base_dir is a per-simulation parameter for diversity)."""
    wind_speed = float(np.clip(5.0 + rng.normal(0, 1.5), 1.0, 12.0))

    base_dir   = wind_base_dir + (90.0 / T) * t
    wind_dir   = float((base_dir + rng.normal(0, 10)) % 360)

    humidity   = float(np.clip(60 + 25 * math.sin(math.pi * t / T)
                                + rng.normal(0, 3), 40, 100))

    season     = float(np.clip(0.8 + 0.5 * math.sin(math.pi * t / T), 0.5, 1.5))

    return wind_speed, wind_dir, humidity, season


# ─── GEOMETRY HELPERS (identical to simulate_disease.py) ─────────────────────

def _contact_weight(row_j, col_j, row_k, col_k):
    dr = abs(row_k - row_j)
    dc = abs(col_k - col_j)
    return 1.0 if (dr + dc == 1) else 0.5


def _wind_alignment(row_j, col_j, row_k, col_k, wind_dir_deg):
    dx       = col_k - col_j
    dy       = row_j - row_k
    theta_jk = math.degrees(math.atan2(dy, dx))
    diff     = math.radians(wind_dir_deg - theta_jk)
    return max(0.0, math.cos(diff))


def _spread_prob(sid_k, inf_nbrs, sectors_by_id, wind_dir, humidity, season):
    row_k    = sectors_by_id[sid_k]["row"]
    col_k    = sectors_by_id[sid_k]["col"]
    survival = 1.0
    for j in inf_nbrs:
        row_j = sectors_by_id[j]["row"]
        col_j = sectors_by_id[j]["col"]
        cw    = _contact_weight(row_j, col_j, row_k, col_k)
        ww    = _wind_alignment(row_j, col_j, row_k, col_k, wind_dir)
        p_j   = min((ALPHA * cw + BETA * ww) * (humidity / 100.0) * season, 0.95)
        survival *= (1.0 - p_j)
    return 1.0 - survival


# ─── SINGLE SIMULATION ────────────────────────────────────────────────────────

def run_one_simulation(sectors_by_id, initial_seeds, wind_base_dir, rng):
    """
    Returns a compact (T+1, N_SECTORS) int8 numpy array of true_status.
    Rows = time steps (0 … T), columns = sector_id order (0 … 99).
    """
    status = np.zeros(N_SECTORS, dtype=np.int8)
    timer  = np.zeros(N_SECTORS, dtype=np.int16)

    for sid in initial_seeds:
        status[sid] = 1
        timer[sid]  = HEALING_PERIOD

    result = np.empty((T + 1, N_SECTORS), dtype=np.int8)
    result[0] = status.copy()

    for t in range(1, T + 1):
        _, wd, hum, sea = generate_env(t, T, wind_base_dir, rng)
        new_status = status.copy()
        new_timer  = timer.copy()

        for sid in range(N_SECTORS):
            if status[sid] == 0:
                inf_nbrs = [n for n in sectors_by_id[sid]["neighbors"]
                            if status[n] == 1]
                if inf_nbrs:
                    p = _spread_prob(sid, inf_nbrs, sectors_by_id,
                                     wd, hum, sea)
                    if rng.random() < p:
                        new_status[sid] = 1
                        new_timer[sid]  = HEALING_PERIOD
            elif status[sid] == 1:
                new_timer[sid] -= 1
                if new_timer[sid] <= 0:
                    new_status[sid] = 0
                    new_timer[sid]  = 0

        status        = new_status
        timer         = new_timer
        result[t]     = status.copy()

    return result


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-sims', type=int, default=1000,
                        help='Number of simulations to generate (default: 1000)')
    parser.add_argument('--out-file', type=str,
                        default=os.path.join(ROOT_DIR,
                                             'simulation_scaled', 'dataset.npy'),
                        help='Output path for the .npy dataset file')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)

    import json
    with open(JSON_PATH) as f:
        grid_config = json.load(f)
    sectors_by_id = {s["sector_id"]: s for s in grid_config}

    # Non-corner sector IDs available for initial seeding
    candidate_sids = [sid for sid in range(N_SECTORS)
                      if sid not in _CORNER_SIDS]

    print(f"Generating {args.n_sims} simulations → {args.out_file}")
    print(f"  Grid        : {N_ROWS}×{N_COLS} = {N_SECTORS} sectors")
    print(f"  Days        : {T}")
    print(f"  Init seeds  : 1–3 random non-corner sectors per sim")
    print()

    all_sims = np.empty((args.n_sims, T + 1, N_SECTORS), dtype=np.int8)

    for i in range(args.n_sims):
        rng = np.random.default_rng(seed=i)           # reproducible per-sim

        # ── Diversity parameters ──────────────────────────────────────────────
        n_seeds      = int(rng.integers(1, 4))         # 1–3 initial infections
        init_seeds   = list(map(int,
                           rng.choice(candidate_sids, size=n_seeds,
                                      replace=False)))
        wind_base    = float(rng.uniform(0, 360))      # random wind direction

        # ── Run simulation ────────────────────────────────────────────────────
        all_sims[i] = run_one_simulation(sectors_by_id, init_seeds,
                                         wind_base, rng)

        # ── Progress ──────────────────────────────────────────────────────────
        if (i + 1) % 100 == 0 or i == args.n_sims - 1:
            pct        = (i + 1) / args.n_sims * 100
            peak_inf   = int(all_sims[i].sum(axis=1).max())
            bar_filled = int(pct / 5)
            bar        = "#" * bar_filled + "-" * (20 - bar_filled)
            print(f"  [{bar}] {i+1:>5}/{args.n_sims}  {pct:5.1f}%"
                  f"  seeds={n_seeds} wind={wind_base:5.1f}°"
                  f"  peak_inf={peak_inf:3d}")
            sys.stdout.flush()

    # ── Save as single .npy file ──────────────────────────────────────────────
    np.save(args.out_file, all_sims)
    size_mb = all_sims.nbytes / 1_000_000
    print(f"\nDone. Saved {args.n_sims} simulations to {args.out_file}")
    print(f"  Shape : {all_sims.shape}  dtype={all_sims.dtype}")
    print(f"  Size  : {size_mb:.1f} MB")


if __name__ == "__main__":
    main()

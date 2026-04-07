"""
simulate_disease.py  (scaled)

Simulates Rice Blast (Pyricularia oryzae) spread across a 10×10
grid of paddy sectors over 72 days.

Spread model (same equations as original, tuned for larger grid):
    spread_prob = 1 - Π(1 - p_j)  for all infected neighbours j
    p_j = (α × contact_weight(j,k) + β × wind_alignment(j,k))
          × humidity_factor
          × season_multiplier

Disease control:
    Infected sectors heal after HEALING_PERIOD days.
    All updates applied simultaneously (no chain reactions).

Output:
    simulation_scaled/simulation_log.csv
    (7,300 rows: 100 sectors × 73 time steps — day 0 through day 72)
"""

import os
import json
import math
import random
import numpy as np
import pandas as pd

# ─── CONSTANTS ────────────────────────────────────────────────────────────────

T              = 72    # total simulation days
HEALING_PERIOD = 20     # days for a sector to self-heal WITHOUT UAV treatment.
                       # With active UAV monitoring, treatment takes only
                       # TREATMENT_DAYS=3 days (defined in uav_env.py), making
                       # UAV detection ~2.3× faster than natural recovery.
ALPHA          = 0.015  # base contact spread weight  (tuned for ~25 infected/day)
BETA           = 0.03  # wind-driven spread weight
RANDOM_SEED    = 42

# ─── PATHS ────────────────────────────────────────────────────────────────────

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
ROOT_DIR    = os.path.dirname(PROJECT_DIR)
GRID_DIR    = os.path.join(ROOT_DIR, "grid_scaled")
OUTPUT_DIR  = os.path.join(ROOT_DIR, "simulation_scaled")

CSV_PATH    = os.path.join(GRID_DIR, "sector_status.csv")
JSON_PATH   = os.path.join(GRID_DIR, "grid_config.json")


# ─── ENVIRONMENTAL VARIABLE GENERATOR ────────────────────────────────────────

def generate_env(t, T):
    """
    Generates synthetic environmental variables for day t.

      wind_speed     : base 5 m/s with Gaussian noise, clipped [1, 12]
      wind_direction : slowly drifts 45° → 135° over the simulation
      humidity       : sinusoidal, peaks ~85% at day 36 (midpoint)
      season_mult    : bell curve, peaks 1.3× at midpoint
    """
    wind_speed = float(np.clip(5.0 + np.random.normal(0, 1.5), 1.0, 12.0))

    base_dir   = 45 + (90 / T) * t
    wind_dir   = float((base_dir + np.random.normal(0, 10)) % 360)

    humidity   = float(np.clip(55 + 20 * math.sin(math.pi * t / T)
                               + np.random.normal(0, 3), 40, 85))

    season     = float(np.clip(0.8 + 0.4 * math.sin(math.pi * t / T), 0.5, 1.2))

    return wind_speed, wind_dir, humidity, season


# ─── GEOMETRY HELPERS ─────────────────────────────────────────────────────────

def get_contact_weight(row_j, col_j, row_k, col_k):
    """1.0 for edge-sharing neighbours, 0.5 for diagonal neighbours."""
    dr = abs(row_k - row_j)
    dc = abs(col_k - col_j)
    return 1.0 if (dr + dc == 1) else 0.5


def get_wind_alignment(row_j, col_j, row_k, col_k, wind_dir_deg):
    """
    Fraction of wind blowing FROM j TOWARD k.
    wind_alignment = max(0, cos(θ_wind − θ_jk))
    """
    dx       = col_k - col_j
    dy       = row_j - row_k   # north = negative row
    theta_jk = math.degrees(math.atan2(dy, dx))
    diff     = math.radians(wind_dir_deg - theta_jk)
    return max(0.0, math.cos(diff))


# ─── SPREAD PROBABILITY ───────────────────────────────────────────────────────

def compute_spread_prob(sector_k, infected_neighbors, sectors_by_id,
                        wind_dir, humidity, season_mult):
    """
    P(sector k gets infected this step) = 1 − Π(1 − p_j) over infected neighbours.
    """
    row_k    = sectors_by_id[sector_k]["row"]
    col_k    = sectors_by_id[sector_k]["col"]
    survival = 1.0

    for j in infected_neighbors:
        row_j     = sectors_by_id[j]["row"]
        col_j     = sectors_by_id[j]["col"]
        contact_w = get_contact_weight(row_j, col_j, row_k, col_k)
        wind_w    = get_wind_alignment(row_j, col_j, row_k, col_k, wind_dir)
        p_j       = min((ALPHA * contact_w + BETA * wind_w)
                        * (humidity / 100.0) * season_mult, 0.95)
        survival *= (1.0 - p_j)

    return 1.0 - survival


# ─── MAIN SIMULATION ──────────────────────────────────────────────────────────

def run_simulation():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load initial state ────────────────────────────────────────────────────
    df_init       = pd.read_csv(CSV_PATH)
    with open(JSON_PATH) as f:
        grid_config = json.load(f)

    sectors_by_id = {s["sector_id"]: s for s in grid_config}
    N_SECTORS     = len(sectors_by_id)

    status = {int(row.sector_id): int(row.true_status)
              for _, row in df_init.iterrows()}
    timer  = {sid: (HEALING_PERIOD if st == 1 else 0)
              for sid, st in status.items()}

    # ── Helper: record one time step ──────────────────────────────────────────
    def record_step(t, wind_speed, wind_dir, humidity, season):
        rows = []
        for sid, s in sectors_by_id.items():
            rows.append({
                "time_step":         t,
                "sector_id":         sid,
                "row":               s["row"],
                "col":               s["col"],
                "x":                 s["x"],
                "y":                 s["y"],
                "true_status":       status[sid],
                "label":             "infected" if status[sid] == 1 else "healthy",
                "treatment_timer":   timer[sid],
                "wind_speed":        round(wind_speed, 2),
                "wind_direction":    round(wind_dir, 2),
                "humidity":          round(humidity, 2),
                "season_multiplier": round(season, 3),
                "neighbors":         ",".join(str(n) for n in s["neighbors"]),
            })
        return rows

    # ── t = 0 ─────────────────────────────────────────────────────────────────
    ws0, wd0, hum0, sea0 = generate_env(0, T)
    records = record_step(0, ws0, wd0, hum0, sea0)

    # ── Days 1 → T ────────────────────────────────────────────────────────────
    for t in range(1, T + 1):
        ws, wd, hum, sea = generate_env(t, T)

        new_status = dict(status)
        new_timer  = dict(timer)

        for sid, s in sectors_by_id.items():
            if status[sid] == 0:
                inf_nbrs = [n for n in s["neighbors"] if status[n] == 1]
                if inf_nbrs:
                    p = compute_spread_prob(sid, inf_nbrs, sectors_by_id,
                                           wd, hum, sea)
                    if random.random() < p:
                        new_status[sid] = 1
                        new_timer[sid]  = HEALING_PERIOD

            elif status[sid] == 1:
                new_timer[sid] -= 1
                if new_timer[sid] <= 0:
                    new_status[sid] = 0
                    new_timer[sid]  = 0

        status  = new_status
        timer   = new_timer
        records += record_step(t, ws, wd, hum, sea)

    # ── Save ──────────────────────────────────────────────────────────────────
    df_out   = pd.DataFrame(records)
    out_path = os.path.join(OUTPUT_DIR, "simulation_log.csv")
    df_out.to_csv(out_path, index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"Simulation complete  →  {T} days  |  {N_SECTORS} sectors  |  "
          f"{len(df_out)} total rows")
    print(f"Output: {out_path}\n")
    print(f"{'Day':>4}  {'#Infected':>9}  {'#Healthy':>8}  "
          f"{'Humidity':>8}  {'WindDir':>8}")
    print("─" * 46)
    for t in range(0, T + 1, 6):   # print every 6 days to keep output manageable
        day_df   = df_out[df_out.time_step == t]
        n_inf    = int((day_df.true_status == 1).sum())
        n_hlt    = int((day_df.true_status == 0).sum())
        hum      = day_df.humidity.iloc[0]
        wdir     = day_df.wind_direction.iloc[0]
        print(f"{t:>4}  {n_inf:>9}  {n_hlt:>8}  {hum:>7.1f}%  {wdir:>7.1f}°")
    print("─" * 46)

    max_inf = max(
        int((df_out[df_out.time_step == t].true_status == 1).sum())
        for t in range(T + 1)
    )
    print(f"\nPeak simultaneous infections: {max_inf} / {N_SECTORS} sectors")


if __name__ == "__main__":
    run_simulation()

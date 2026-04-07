"""
build_grid.py  (scaled)

Builds the 10×10 field grid (100 sectors) for the scaled-up
Multi-UAV Active Sensing simulation.

No image generation — dataset-agnostic.
Assigns initial true_status (healthy / infected) and records
8-connected neighbour lists.

Outputs (relative to project root):
  grid_scaled/
  ├── grid_config.json        (grid structure + neighbours)
  └── sector_status.csv       (sector labels at t=0)

Status codes:
  0 = healthy
  1 = infected
  2 = unknown  (uav_status starts as 2 — UAVs haven't visited yet)
"""

import os
import json
import pandas as pd

# ─── CONFIG ───────────────────────────────────────────────────────────────────

GRID_ROWS     = 10
GRID_COLS     = 10

# Three geographically spread seed infections at t=0:
#   sector 22  →  row 2, col 2  (upper-left quadrant)
#   sector 55  →  row 5, col 5  (centre)
#   sector 77  →  row 7, col 7  (lower-right quadrant)
SEED_INFECTED = [22, 55, 77]

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR   = os.path.dirname(SCRIPT_DIR)                       # .../project/
ROOT_DIR      = os.path.dirname(PROJECT_DIR)                      # .../Exploratory Project/
OUTPUT_DIR    = os.path.join(ROOT_DIR, "grid_scaled")


# ─── GRID BUILDER ─────────────────────────────────────────────────────────────

def build_sector_grid(rows, cols):
    """
    Creates list of sector dicts with ID, (row, col) position,
    and 8-connected neighbour list.

    sector_id = row * cols + col
    """
    sectors = []
    for r in range(rows):
        for c in range(cols):
            sid       = r * cols + c
            neighbors = []
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        neighbors.append(nr * cols + nc)
            sectors.append({
                "sector_id": sid,
                "row":       r,
                "col":       c,
                "x":         float(c),
                "y":         float(r),
                "neighbors": neighbors,
            })
    return sectors


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def build_grid():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    sectors = build_sector_grid(GRID_ROWS, GRID_COLS)
    n_total = len(sectors)

    records = []
    for s in sectors:
        sid         = s["sector_id"]
        true_status = 1 if sid in SEED_INFECTED else 0
        label       = "infected" if true_status == 1 else "healthy"

        records.append({
            "sector_id":   sid,
            "row":         s["row"],
            "col":         s["col"],
            "x":           s["x"],
            "y":           s["y"],
            "neighbors":   ",".join(str(n) for n in s["neighbors"]),
            "true_status": true_status,
            "uav_status":  2,
            "label":       label,
            "time_step":   0,
        })

    # Save sector status CSV
    df       = pd.DataFrame(records)
    csv_path = os.path.join(OUTPUT_DIR, "sector_status.csv")
    df.to_csv(csv_path, index=False)

    # Save grid config JSON
    json_path = os.path.join(OUTPUT_DIR, "grid_config.json")
    with open(json_path, "w") as f:
        json.dump(sectors, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    n_infected = len(SEED_INFECTED)
    n_healthy  = n_total - n_infected

    print(f"Grid  →  {GRID_ROWS} rows × {GRID_COLS} cols = {n_total} sectors")
    print(f"Seed infected : {SEED_INFECTED}")
    print(f"  Healthy at t=0  : {n_healthy}")
    print(f"  Infected at t=0 : {n_infected}")
    print()
    print(f"Outputs saved to: {OUTPUT_DIR}/")
    print(f"  sector_status.csv  →  {csv_path}")
    print(f"  grid_config.json   →  {json_path}")

    # Print a compact grid map
    print()
    print("Grid layout (I=infected, .=healthy):")
    print("     " + "".join(f"{c:3}" for c in range(GRID_COLS)))
    for r in range(GRID_ROWS):
        row_str = f"r{r:2}  "
        for c in range(GRID_COLS):
            sid = r * GRID_COLS + c
            row_str += "  I" if sid in SEED_INFECTED else "  ."
        print(row_str)


if __name__ == "__main__":
    build_grid()

"""
Example runs of the PtM scheduling MILP on the MasCOR dataset.

Cases:
    1. MasCOR synthetic 576-h scenario (Skive, index 0) with its sampled design,
       flexible vs. rigid (constant nameplate) methanol synthesis.
    2. Historical year 2023 (Skive, 8760 h) with a demonstration design.

Each case is solved as MILP and as LP relaxation (RMIP) to report the
integrality gap and solve times. Hourly results are written to ./results.
"""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from ptm_scheduling_gamspy import (
    OperatingParams,
    PlantDesign,
    SchedulingData,
    build_model,
    hourly_results,
    kpis,
    solve,
)

# Path to the MasCOR Kaggle "dataset/" folder; override with MASCOR_DATASET.
MASCOR_DATA = Path(os.environ.get("MASCOR_DATASET", Path(__file__).resolve().parents[1] / "mascor" / "dataset"))
RESULTS = Path(__file__).resolve().parent / "results"

SP_H2_RATED = 55.7  # kWh/kg H2, MasCOR PEM sizing
GAMMA_H2 = 0.19576  # kg H2/kg MeOH


def wind_power_cf(w, w_in: float = 1.5, w_rated: float = 12.0, w_out: float = 25.0) -> np.ndarray:
    """MasCOR turbine curve (SI Eq. 38) as capacity factor."""

    w = np.asarray(w, dtype=float)
    cf = np.where((w > w_in) & (w <= w_rated), (w**3 - w_in**3) / (w_rated**3 - w_in**3), 0.0)
    return np.where((w > w_rated) & (w <= w_out), 1.0, cf)


def load_mascor_instance(country: str, region: str, idx: int, eps: float = 0.5):
    """Synthetic scenario and sampled design from the MasCOR oracle inputs."""

    od = MASCOR_DATA / country / region / "oracle_dataset_c_fax_fix_sample_50000"
    wind = np.load(od / "wind_scenario_seed2024.npy", mmap_mode="r")[idx]
    price = np.load(od / "price_scenario_seed2024.npy", mmap_mode="r")[idx] * 1000.0  # USD/kWh -> USD/MWh
    x_flow, lh2_cap, ess_kwh, pem_ratio, _ = np.load(od / "design_spec_seed2024.npy", mmap_mode="r")[idx]
    p_min = x_flow * GAMMA_H2 * SP_H2_RATED / 1000.0
    p_max = p_min + lh2_cap * SP_H2_RATED / 1000.0
    design = PlantDesign(
        C_wind_MW=50.0,
        C_PEM_MW=float(pem_ratio * (p_max - p_min) + p_min),
        C_BESS_MWh=float(ess_kwh / 1000.0),
        C_H2_kg=float(lh2_cap),
        m_MeOH_nom_kgph=float(x_flow),
        G_max_MW=100.0,
    )
    return SchedulingData.from_arrays(wind, price, eps), design


def load_historical_year(country: str, region: str, year: int, row: int = 0, eps: float = 0.5) -> SchedulingData:
    """Historical 50-m wind speed and day-ahead price (negative prices kept)."""

    w = np.load(MASCOR_DATA / country / region / "weather_unique_data.npy")[row]
    prices = pd.read_csv(MASCOR_DATA / "european_electricity_price_data_hourly" / f"{country}.csv")
    prices = prices.iloc[: len(w)]
    sel = (pd.to_datetime(prices["Datetime (UTC)"]).dt.year == year).to_numpy()
    price = prices["Price (EUR/MWhe)"].to_numpy(dtype=float)[sel] * 1.03  # EUR -> USD as in MasCOR
    return SchedulingData.from_arrays(wind_power_cf(w[sel]), price, eps)


def run_case(name: str, data: SchedulingData, design: PlantDesign, par: OperatingParams, time_limit: float) -> dict:
    m, model, relaxed, s = build_model(data, design, par)
    t_mip, summary = solve(model, time_limit=time_limit)
    obj_mip = model.objective_value
    df = hourly_results(data, design, par, s)
    res = kpis(df, design, par)
    res["shortfall_t"] = float(s["short"].toValue() or 0.0)
    t_lp, _ = solve(relaxed, time_limit=time_limit)
    obj_lp = relaxed.objective_value

    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / f"{name}_hourly.csv", index=False)

    row = summary.iloc[0]
    print(f"\n=== {name}: {len(data.time_steps)} h ===")
    print(
        f"  equations {row['Num of Equations']}, variables {row['Num of Variables']}, "
        f"status {row['Model Status']} / {row['Solver Status']}"
    )
    print(f"  MILP objective {obj_mip:,.0f} USD in {t_mip:.1f} s")
    print(f"  RMIP objective {obj_lp:,.0f} USD in {t_lp:.1f} s")
    # Absolute gap: the objective is a net cost and can be close to zero.
    print(f"  integrality gap {obj_mip - obj_lp:,.0f} USD")
    for key, val in res.items():
        print(f"  {key:>20s}: {val:,.3f}")
    return {"case": name, "obj_MILP": obj_mip, "obj_RMIP": obj_lp, "t_MILP_s": t_mip, "t_RMIP_s": t_lp, **res}


def main() -> None:
    base = OperatingParams()
    # Same 90 % production target; rigid synthesis is either off or at nameplate.
    rigid = replace(base, f_min_MeOH=1.0)
    rows = []

    data, design = load_mascor_instance("Denmark", "Skive", idx=0)
    print("MasCOR design (Skive, idx 0):", design)
    rows.append(run_case("mascor576_flexible", data, design, base, time_limit=300))
    rows.append(run_case("mascor576_rigid", data, design, rigid, time_limit=300))

    year_data = load_historical_year("Denmark", "Skive", 2023)
    demo = PlantDesign(C_wind_MW=50.0, C_PEM_MW=20.0, C_BESS_MWh=20.0, C_H2_kg=1000.0, m_MeOH_nom_kgph=1000.0, G_max_MW=50.0)
    print("\nDemo design (historical 2023):", demo)
    rows.append(run_case("skive2023_flexible", year_data, demo, base, time_limit=900))

    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS / "summary.csv", index=False)
    print("\n", summary[["case", "obj_MILP", "obj_RMIP", "t_MILP_s", "t_RMIP_s", "MeOH_load_factor", "CI_tCO2_per_tMeOH"]])


if __name__ == "__main__":
    sys.exit(main())

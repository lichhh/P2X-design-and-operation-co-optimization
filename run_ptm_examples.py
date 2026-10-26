"""
Example runs of the PtM scheduling MILP on the MasCOR dataset.

Cases:
    1. MasCOR synthetic 576-h scenario (Skive, index 0) with its sampled design,
       wind only, methanol nameplate equal to the requirement (kappa = 1).
    2. The same scenario with 25 % oversized methanol synthesis (kappa = 1.25).
    3. Historical year 2023 at Skive (8760 h): wind + PV, hourly DK1 grid
       emission factor, demonstration design with kappa = 1.25.

Each case is solved as MILP and as LP relaxation (RMIP) to report the
integrality gap and solve times. Hourly results are written to ./results.

External inputs, downloaded once and cached in ./data:
    - PV capacity factor: PVGIS v5.3 hourly series (EU JRC, SARAH3 radiation),
      1 kWp c-Si, optimal fixed tilt and azimuth, 14 % system loss.
    - Grid emission factor: Energinet dataset "Declaration Grid Emission",
      125 % CHP fuel-allocation method, fossil CO2 per kWh.
"""

from __future__ import annotations

import json
import os
import ssl
import sys
import urllib.parse
import urllib.request
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
DATA = Path(__file__).resolve().parent / "data"
RESULTS = Path(__file__).resolve().parent / "results"

SKIVE_LAT, SKIVE_LON = 56.57, 9.03  # GreenLab Skive, approximate
SP_H2_RATED = 55.7  # kWh/kg H2, MasCOR PEM sizing
GAMMA_H2 = 0.19576  # kg H2/kg MeOH
EUR_TO_USD = 1.03  # as in MasCOR


def wind_power_cf(w, w_in: float = 1.5, w_rated: float = 12.0, w_out: float = 25.0) -> np.ndarray:
    """MasCOR turbine curve (SI Eq. 38) as capacity factor."""

    w = np.asarray(w, dtype=float)
    cf = np.where((w > w_in) & (w <= w_rated), (w**3 - w_in**3) / (w_rated**3 - w_in**3), 0.0)
    return np.where((w > w_rated) & (w <= w_out), 1.0, cf)


def _get_json(url: str):
    try:
        import certifi

        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
    with urllib.request.urlopen(url, timeout=180, context=ctx) as r:
        return json.loads(r.read().decode())


def pv_capacity_factor(year: int, lat: float = SKIVE_LAT, lon: float = SKIVE_LON) -> pd.Series:
    """Hourly PV capacity factor [-] indexed by UTC hour (PVGIS v5.3)."""

    path = DATA / f"pvgis_cf_{lat}_{lon}_{year}.csv"
    if not path.exists():
        url = (
            "https://re.jrc.ec.europa.eu/api/v5_3/seriescalc?"
            f"lat={lat}&lon={lon}&pvcalculation=1&peakpower=1&loss=14"
            f"&startyear={year}&endyear={year}&optimalangles=1&outputformat=json"
        )
        rows = _get_json(url)["outputs"]["hourly"]
        df = pd.DataFrame(
            {
                "time_utc": pd.to_datetime([r["time"][:11] for r in rows], format="%Y%m%d:%H"),
                "cf": [r["P"] / 1000.0 for r in rows],  # W per kWp -> capacity factor
            }
        )
        DATA.mkdir(exist_ok=True)
        df.to_csv(path, index=False)
    return pd.read_csv(path, parse_dates=["time_utc"]).set_index("time_utc")["cf"]


def dk_grid_emission(year: int, area: str = "DK1", method: str = "125%") -> pd.Series:
    """Hourly grid emission factor [tCO2/MWh] indexed by UTC hour (Energinet)."""

    path = DATA / f"energinet_co2_{area}_{method.rstrip('%')}_{year}.csv"
    if not path.exists():
        flt = urllib.parse.quote(json.dumps({"PriceArea": [area], "FuelAllocationMethod": [method]}))
        url = (
            "https://api.energidataservice.dk/dataset/DeclarationGridEmission?"
            f"start={year - 1}-12-31T00:00&end={year + 1}-01-02T00:00&filter={flt}&limit=0"
        )
        recs = _get_json(url)["records"]
        df = pd.DataFrame(
            {
                "time_utc": pd.to_datetime([r["HourUTC"] for r in recs]),
                "eps": [r["CO2PerkWh"] / 1000.0 for r in recs],  # g/kWh -> t/MWh
            }
        )
        df = df[df["time_utc"].dt.year == year].sort_values("time_utc")
        DATA.mkdir(exist_ok=True)
        df.to_csv(path, index=False)
    return pd.read_csv(path, parse_dates=["time_utc"]).set_index("time_utc")["eps"]


def _align(series: pd.Series, index: pd.DatetimeIndex, name: str) -> np.ndarray:
    aligned = series[~series.index.duplicated()].reindex(index)
    missing = int(aligned.isna().sum())
    if missing:
        print(f"  {name}: {missing} missing hours filled by interpolation")
    return aligned.interpolate(limit_direction="both").to_numpy()


def load_mascor_instance(country: str, region: str, idx: int, kappa: float, eps: float):
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
        m_MeOH_nom_kgph=float(kappa * x_flow),
        D_MeOH_kgph=float(x_flow),
        G_max_MW=100.0,
    )
    return SchedulingData.from_arrays(wind, price, eps), design


def load_historical_year(country: str, region: str, year: int, row: int = 0) -> SchedulingData:
    """Historical 50-m wind speed, day-ahead price (negative prices kept), PV and grid emissions."""

    w = np.load(MASCOR_DATA / country / region / "weather_unique_data.npy")[row]
    prices = pd.read_csv(MASCOR_DATA / "european_electricity_price_data_hourly" / f"{country}.csv")
    prices = prices.iloc[: len(w)]
    stamps = pd.to_datetime(prices["Datetime (UTC)"])
    sel = (stamps.dt.year == year).to_numpy()
    index = pd.DatetimeIndex(stamps[sel])
    price = prices["Price (EUR/MWhe)"].to_numpy(dtype=float)[sel] * EUR_TO_USD
    cf_pv = _align(pv_capacity_factor(year), index, "PV")
    eps = _align(dk_grid_emission(year), index, "grid emission")
    return SchedulingData.from_arrays(wind_power_cf(w[sel]), price, eps, cf_pv)


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
    print(f"\n=== {name}: {len(data.time_steps)} h, kappa = {design.oversize:.2f} ===")
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
    return {
        "case": name,
        "equations": row["Num of Equations"],
        "variables": row["Num of Variables"],
        "obj_MILP": obj_mip,
        "obj_RMIP": obj_lp,
        "t_MILP_s": t_mip,
        "t_RMIP_s": t_lp,
        **res,
    }


def main() -> None:
    par = OperatingParams()
    rows = []

    year_data = load_historical_year("Denmark", "Skive", 2023)
    eps_mean = float(np.mean(year_data.eps_grid))
    print(f"DK1 2023 mean grid emission factor: {eps_mean:.4f} tCO2/MWh; PV CF mean {np.mean(year_data.cf_pv):.3f}")

    # MasCOR scenarios carry no timestamps, so they use the 2023 DK1 mean.
    for name, kappa in (("mascor576_k1.00", 1.0), ("mascor576_k1.25", 1.25)):
        data, design = load_mascor_instance("Denmark", "Skive", idx=0, kappa=kappa, eps=eps_mean)
        print(f"\n{name}: {design}")
        rows.append(run_case(name, data, design, par, time_limit=300))

    demo = PlantDesign(
        C_wind_MW=50.0,
        C_PV_MW=25.0,
        C_PEM_MW=20.0,
        C_BESS_MWh=20.0,
        C_H2_kg=1000.0,
        m_MeOH_nom_kgph=1250.0,
        D_MeOH_kgph=1000.0,
        G_max_MW=50.0,
    )
    print(f"\nDemo design (historical 2023): {demo}")
    rows.append(run_case("skive2023_windpv_k1.25", year_data, demo, par, time_limit=900))

    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS / "summary.csv", index=False)
    cols = ["case", "obj_MILP", "obj_RMIP", "t_MILP_s", "MeOH_vs_requirement", "MeOH_load_factor", "CI_tCO2_per_tMeOH"]
    print("\n", summary[cols].to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())

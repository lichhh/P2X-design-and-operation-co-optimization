"""
Full-plant runs with the explicit downstream section: crude methanol tank,
FVC distillation column and a thermal store fed by the synthesis heat.

Historical year 2023 at Skive (wind + PV, DK1 day-ahead prices, hourly grid
emission factor), demonstration design of run_ptm_examples.py with kappa = 1.25.
Cases (finite differences give a first view of the value of each unit):
    lumped          distillation inside SP_MeOH, horizon requirement (reference)
    fvc_tes         crude tank 24 t, TES 8 MWh_th, FVC compressor 0.4 MW
    no_syn_heat     as fvc_tes, synthesis heat not recovered (q_syn = 0)
    eboiler_tes     no FVC; 1 MW electric boiler charges the TES
    fvc_no_tes      no thermal store (FVC runs whenever synthesis heat is short)
    no_crude_tank   synthesis locked to the constant column load
    match_annual    fvc_tes with annual matching: import <= 70 % of load,
                    export + curtailment <= 40 % of available generation

Usage: python run_fvc_examples.py [hours]   (default 8760; e.g. 672 for a quick test)
"""

from __future__ import annotations

import sys
from dataclasses import replace

import numpy as np
import pandas as pd

from ptm_scheduling_gamspy import OperatingParams, PlantDesign, SchedulingData, build_model, hourly_results, kpis, solve
from run_ptm_examples import RESULTS, load_historical_year

DEMO = PlantDesign(
    C_wind_MW=50.0,
    C_PV_MW=25.0,
    C_PEM_MW=20.0,
    C_BESS_MWh=20.0,
    C_H2_kg=1000.0,
    m_MeOH_nom_kgph=1250.0,
    D_MeOH_kgph=1000.0,
    G_max_MW=50.0,
    C_crude_kg=24000.0,
    E_TES_MWh=8.0,
    W_FVC_MW=0.40,
)
EXPLICIT = OperatingParams(distillation="explicit")

CASES = {
    "lumped": (DEMO, OperatingParams()),
    "fvc_tes": (DEMO, EXPLICIT),
    "no_syn_heat": (DEMO, replace(EXPLICIT, q_syn=0.0)),
    "eboiler_tes": (replace(DEMO, W_FVC_MW=0.0, C_eboiler_MW=1.0), EXPLICIT),
    "fvc_no_tes": (replace(DEMO, E_TES_MWh=0.0), EXPLICIT),
    "no_crude_tank": (replace(DEMO, C_crude_kg=0.0), EXPLICIT),
    "match_annual": (DEMO, replace(EXPLICIT, match_window_h=8760, phi_import=0.7, phi_spill=0.4)),
}


def window(data: SchedulingData, n: int) -> SchedulingData:
    return SchedulingData(
        data.time_steps[:n], data.cf_wind[:n], data.price[:n], data.eps_grid[:n], data.cf_pv[:n]
    )


def run(name: str, data: SchedulingData, design: PlantDesign, par: OperatingParams) -> dict:
    m, model, relaxed, s = build_model(data, design, par)
    wall, summary = solve(model, gap=1e-4, time_limit=1800)
    df = hourly_results(data, design, par, s)
    df.to_csv(RESULTS / f"fvc_{name}_hourly.csv", index=False)
    res = kpis(df, design, par)
    row = summary.iloc[0]
    print(f"\n=== {name}: {len(data.time_steps)} h ===")
    print(f"  equations {row['Num of Equations']}, variables {row['Num of Variables']}, "
          f"status {row['Model Status']} / {row['Solver Status']}, {wall:.1f} s")
    print(f"  objective {model.objective_value:,.0f} USD")
    keys = ("MeOH_vs_requirement", "MeOH_load_factor", "crude_swing_t", "FVC_on_h", "FVC_MWh", "eboiler_MWh",
            "dist_kWh_per_kg", "syn_heat_MWh", "TES_release_MWh", "heat_dumped_MWh", "TES_full_cycles",
            "import_MWh", "export_MWh", "curtail_MWh")
    for key in keys:
        if key in res:
            print(f"  {key:>20s}: {res[key]:,.3f}")
    return {"case": name, "obj_USD": model.objective_value, "solve_s": wall, "status": row["Model Status"], **res}


def main() -> None:
    hours = int(sys.argv[1]) if len(sys.argv) > 1 else 8760
    data = window(load_historical_year("Denmark", "Skive", 2023), hours)
    RESULTS.mkdir(exist_ok=True)
    rows = [run(name, data, design, par) for name, (design, par) in CASES.items()]
    summary = pd.DataFrame(rows)
    ref = summary.set_index("case")["obj_USD"]["fvc_tes"]
    summary["delta_vs_fvc_tes_USD"] = summary["obj_USD"] - ref
    summary.to_csv(RESULTS / f"fvc_summary_{hours}h.csv", index=False)
    cols = ["case", "obj_USD", "delta_vs_fvc_tes_USD", "solve_s", "FVC_on_h", "dist_kWh_per_kg", "heat_dumped_MWh"]
    print("\n", summary[[c for c in cols if c in summary]].to_string(index=False, float_format=lambda x: f"{x:,.1f}"))


if __name__ == "__main__":
    sys.exit(main())

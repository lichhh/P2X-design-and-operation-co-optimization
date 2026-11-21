"""
Decision-based evaluation of an ensemble of generated scenarios (off-grid PtM).

Every weather series X is used to design the plant, and the design is then operated
on the real 2024-2025 series. Regret is the extra annual cost per tonne of methanol
relative to the design made on the real series.

Series:
    real     observed 2024-2025 wind (NASA POWER WS50M) and GHI
    s01-s50  generated scenarios
    hYYYY    observed two-year blocks of other years from NASA POWER, the design
             basis available without a generator; they do not know the 2024-2025
             window means that condition the generated scenarios

Consensus designs average the capacities of K scenario designs. Averaging removes
most of the sampling noise, so the regret left at K = 50 is mainly systematic.
The K = 50 design is also operated on every scenario, which gives a predicted cost
distribution to compare with its cost on the real series.

Usage:
    python evaluate_ensemble.py prepare --data <folder with scenario_XX.csv>
    python evaluate_ensemble.py design --tag k125 --kappa 1.25
    python evaluate_ensemble.py consensus --tag k125 --kappa 1.25
    python evaluate_ensemble.py report --tag k125
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_scenarios import OUT, PV_PR
from offgrid_design_gamspy import BARRIER, Capacities, OffgridCase, capacities_of, design_lp, solve_offgrid
from run_ptm_examples import DATA, _get_json, wind_power_cf

INPUTS = OUT / "ensemble_inputs.npz"
SCENARIOS = [f"s{i:02d}" for i in range(1, 51)]
BLOCKS = [(2005, 2006), (2007, 2008), (2009, 2010), (2011, 2012), (2013, 2014), (2015, 2016), (2017, 2018), (2019, 2020), (2022, 2023)]
HISTORY = [f"h{a}" for a, _ in BLOCKS]
SITE_LAT, SITE_LON = 55.0, 9.25  # NASA POWER cell that reproduces the real series exactly
SUBSET_SIZES = (2, 5, 10, 20)
SUBSET_REPEATS = 10
CAP_COLS = ["C_wind_MW", "C_PV_MW", "N_PEM", "C_BESS_MWh", "C_H2_kg"]
EARLIER_RUN = {"k125": "", "k100": "_k100", "k150": "_k150", "pvheavy": "_pvheavy"}


def power_year(year: int) -> pd.DataFrame:
    path = DATA / f"nasa_power_{SITE_LAT}_{SITE_LON}_{year}.csv"
    if not path.exists():
        url = (
            "https://power.larc.nasa.gov/api/temporal/hourly/point?parameters=WS50M,ALLSKY_SFC_SW_DWN&community=RE"
            f"&longitude={SITE_LON}&latitude={SITE_LAT}&start={year}0101&end={year}1231&format=JSON&time-standard=UTC"
        )
        for attempt in range(5):
            try:
                par = _get_json(url)["properties"]["parameter"]
                break
            except Exception:  # noqa: BLE001  (transient API errors)
                if attempt == 4:
                    raise
                time.sleep(15 * (attempt + 1))
        df = pd.DataFrame(par).rename(columns={"WS50M": "wind_ms", "ALLSKY_SFC_SW_DWN": "ghi_wm2"})
        df.index = pd.to_datetime(df.index, format="%Y%m%d%H")
        df.index.name = "time_utc"
        df = df.where(df > -900).interpolate(limit_direction="both")
        DATA.mkdir(exist_ok=True)
        df.to_csv(path)
    return pd.read_csv(path, parse_dates=["time_utc"]).set_index("time_utc")


def prepare(data_dir: str) -> None:
    d = Path(data_dir)
    real = pd.read_csv(d / "real_2024_2025.csv", parse_dates=["time_utc"])
    series = {"real": (real["wind_ms"].to_numpy(float), real["ghi_wm2"].to_numpy(float))}
    for i, name in enumerate(SCENARIOS, 1):
        s = pd.read_csv(d / f"scenario_{i:02d}.csv", parse_dates=["time_utc"])
        if not (s["time_utc"] == real["time_utc"]).all():
            raise ValueError(f"Time stamps of scenario {i} differ from the real series.")
        series[name] = (s["wind_ms"].to_numpy(float), s["ghi_wm2"].to_numpy(float))
    for (a, b), name in zip(BLOCKS, HISTORY):
        h = pd.concat([power_year(a), power_year(b)])
        series[name] = (h["wind_ms"].to_numpy(float), h["ghi_wm2"].to_numpy(float))
    arrays = {}
    for name, (w, g) in series.items():
        arrays[f"cfw_{name}"] = wind_power_cf(w)
        arrays[f"cfp_{name}"] = np.clip(PV_PR * g / 1000.0, 0.0, 1.0)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(INPUTS, **arrays)
    show = ["real", SCENARIOS[0], SCENARIOS[-1], *HISTORY]
    for name in show:
        print(f"{name:6s} hours {len(arrays['cfw_' + name]):6d}  wind CF {arrays['cfw_' + name].mean():.4f}  PV CF {arrays['cfp_' + name].mean():.4f}")


def load_series(name: str):
    with np.load(INPUTS) as z:
        return z[f"cfw_{name}"], z[f"cfp_{name}"]


def design_task(args) -> dict:
    name, kappa, max_wind, threads, limit = args
    case = OffgridCase(kappa=kappa, max_wind_MW=max_wind)
    des = design_lp(*load_series(name), case, time_limit=limit, threads=threads)
    ev = solve_offgrid(
        *load_series("real"), case, fixed=capacities_of(des), integer=False,
        time_limit=limit, threads=threads, solver_options=BARRIER,
    )
    row = {"dataset": name}
    row.update({f"d_{k}": v for k, v in des.items()})
    row.update({f"e_{k}": v for k, v in ev.items() if k not in CAP_COLS + ["C_PEM_MW"]})
    return row


def test_task(args) -> dict:
    key, caps, test_name, kappa, max_wind, threads, limit = args
    case = OffgridCase(kappa=kappa, max_wind_MW=max_wind)
    ev = solve_offgrid(
        *load_series(test_name), case, fixed=Capacities(**caps), integer=False,
        time_limit=limit, threads=threads, solver_options=BARRIER,
    )
    keep = ("TAC_usd_per_t", "fixed_usd_per_t", "penalty_usd_per_t", "curtail_share", "unserved_MWh_y", "shortfall_t_y", "status", "solve_s")
    return {**key, "test": test_name, **caps, **{k: ev[k] for k in keep}}


def _pool(function, tasks, path: Path, workers: int, label) -> None:
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        futures = {ex.submit(function, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)
            if i % 5 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} {label(row)}  {time.time() - t0:.0f} s", flush=True)


def design_path(tag: str) -> Path:
    return OUT / f"ensemble_design_{tag}.csv"


def consensus_path(tag: str) -> Path:
    return OUT / f"ensemble_consensus_{tag}.csv"


def design(tag, kappa, max_wind, workers, threads, limit) -> None:
    path = design_path(tag)
    done = set(pd.read_csv(path)["dataset"]) if path.exists() else set()
    names = [n for n in ["real", *SCENARIOS, *HISTORY] if n not in done]
    print(f"{len(names)} designs to solve, {len(done)} done", flush=True)
    tasks = [(n, kappa, max_wind, threads, limit) for n in names]
    _pool(design_task, tasks, path, workers, lambda r: f"{r['dataset']} TAC on real {r['e_TAC_usd_per_t']:.2f}")


def mean_caps(df: pd.DataFrame) -> dict:
    caps = {c: float(df[f"d_{c}"].mean()) for c in CAP_COLS}
    caps["N_PEM"] = float(max(1, round(caps["N_PEM"])))
    return caps


def consensus(tag, kappa, max_wind, workers, threads, limit) -> None:
    d = pd.read_csv(design_path(tag)).set_index("dataset")
    s = d.loc[SCENARIOS]
    rng = np.random.default_rng(2026)
    tasks = []
    for k in SUBSET_SIZES:
        for r in range(SUBSET_REPEATS):
            pick = rng.choice(SCENARIOS, k, replace=False)
            tasks.append(({"kind": "subset", "K": k, "rep": r}, mean_caps(s.loc[pick]), "real"))
    all_caps = mean_caps(s)
    tasks.append(({"kind": "subset", "K": 50, "rep": 0}, all_caps, "real"))
    tasks += [({"kind": "predict", "K": 50, "rep": 0}, all_caps, n) for n in SCENARIOS]
    tasks.append(({"kind": "predict", "K": 50, "rep": 0}, all_caps, "real"))
    path = consensus_path(tag)
    done = set()
    if path.exists():
        old = pd.read_csv(path)
        done = set(zip(old["kind"], old["K"], old["rep"], old["test"]))
    open_tasks = [
        (key, caps, test, kappa, max_wind, threads, limit)
        for key, caps, test in tasks
        if (key["kind"], key["K"], key["rep"], test) not in done
    ]
    print(f"{len(open_tasks)} consensus tests to solve", flush=True)
    _pool(test_task, open_tasks, path, workers, lambda r: f"{r['kind']} K={r['K']} on {r['test']}: {r['TAC_usd_per_t']:.2f}")


def quant(x) -> str:
    x = np.asarray(x, dtype=float)
    return f"mean {x.mean():8.2f}  sd {x.std(ddof=1):7.2f}  p5 {np.percentile(x, 5):8.2f}  p50 {np.median(x):8.2f}  p95 {np.percentile(x, 95):8.2f}  max {x.max():8.2f}"


def report(tag: str) -> None:
    d = pd.read_csv(design_path(tag)).set_index("dataset")
    print("design status:", d["d_status"].value_counts().to_dict(), "| test:", d["e_status"].value_counts().to_dict(),
          "| median solve s: %.0f / %.0f" % (d["d_solve_s"].median(), d["e_solve_s"].median()))
    best = d.loc["real", "e_TAC_usd_per_t"]
    pct = lambda s: 100.0 * (s / best - 1.0)  # noqa: E731
    s, h = d.loc[SCENARIOS], d.loc[[n for n in HISTORY if n in d.index]]
    print(f"\nReal-optimal cost {best:.2f} USD/t; regret in % of it")
    print("  50 scenarios :", quant(pct(s["e_TAC_usd_per_t"])))
    print("  history      :", quant(pct(h["e_TAC_usd_per_t"])), f"(n={len(h)})")
    prev = OUT / f"offgrid_design{EARLIER_RUN.get(tag, '_none')}.csv"
    if prev.exists():
        p = pd.read_csv(prev, dtype={"design_span": str, "test_span": str})
        p = p[(p["design_span"] == "2y") & (p["test_span"] == "2y")].set_index("dataset")
        a = 100.0 * (p.loc[[f"analog{i}" for i in range(1, 6)], "e_TAC_usd_per_t"] / p.loc["real", "e_TAC_usd_per_t"] - 1.0)
        print("  5 analogs    :", quant(a))
    print("  scenarios with shortfall on real: %d/50, mean %.1f t/y" % ((s["e_shortfall_t_y"] > 1).sum(), s["e_shortfall_t_y"].mean()))
    fixed = s["e_fixed_usd_per_t"] - d.loc["real", "e_fixed_usd_per_t"]
    pen = s["e_penalty_usd_per_t"] - d.loc["real", "e_penalty_usd_per_t"]
    print("  mean regret split, USD/t: fixed cost %+.2f, penalties %+.2f" % (fixed.mean(), pen.mean()))

    print("\nDesign optimum: real vs distribution over the 50 scenarios (rank = scenarios below real)")
    for c, lab in [("TAC_usd_per_t", "own optimal cost USD/t"), ("C_wind_MW", "wind MW"), ("C_PV_MW", "PV MW"), ("C_PEM_MW", "PEM MW"), ("C_BESS_MWh", "battery MWh"), ("C_H2_kg", "H2 tank kg")]:
        x, e = d.loc["real", f"d_{c}"], s[f"d_{c}"].to_numpy()
        print(f"  {lab:24s} real {x:10.2f} | mean {e.mean():10.2f}  se {e.std(ddof=1) / np.sqrt(50):8.2f}  p5 {np.percentile(e, 5):10.2f}  p95 {np.percentile(e, 95):10.2f} | rank {(e < x).sum():2d}/50")

    if not consensus_path(tag).exists():
        return
    c = pd.read_csv(consensus_path(tag))
    sub = c[c["kind"] == "subset"]
    print("\nConsensus design (mean capacities of K scenario designs), regret on real in %")
    print(f"  K= 1  mean {pct(s['e_TAC_usd_per_t']).mean():6.3f}  sd {pct(s['e_TAC_usd_per_t']).std(ddof=1):6.3f}  (n=50)")
    for k, g in sub.groupby("K"):
        r = pct(g["TAC_usd_per_t"])
        print(f"  K={k:2d}  mean {r.mean():6.3f}  sd {(r.std(ddof=1) if len(r) > 1 else 0):6.3f}  (n={len(r)})")
    pr = c[c["kind"] == "predict"]
    e, x = pr[pr["test"] != "real"], pr[pr["test"] == "real"].iloc[0]
    print("\nK=50 design operated on each scenario vs on real")
    print(f"  capacities: wind {x['C_wind_MW']:.2f} PV {x['C_PV_MW']:.2f} N {x['N_PEM']:.0f} battery {x['C_BESS_MWh']:.2f} H2 {x['C_H2_kg']:.0f}")
    print("  cost on scenarios:", quant(e["TAC_usd_per_t"]))
    print(f"  cost on real {x['TAC_usd_per_t']:.2f}; scenarios cheaper than real {(e['TAC_usd_per_t'] < x['TAC_usd_per_t']).sum()}/50; shortfall on real {x['shortfall_t_y']:.1f} t/y, on scenarios mean {e['shortfall_t_y'].mean():.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["prepare", "design", "consensus", "report"])
    ap.add_argument("--data")
    ap.add_argument("--tag", default="k125")
    ap.add_argument("--kappa", type=float, default=1.25)
    ap.add_argument("--max-wind", type=float, default=500.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=5)
    ap.add_argument("--time-limit", type=float, default=3600.0)
    a = ap.parse_args()
    if a.command == "prepare":
        prepare(a.data)
    elif a.command == "design":
        design(a.tag, a.kappa, a.max_wind, a.workers, a.threads, a.time_limit)
    elif a.command == "consensus":
        consensus(a.tag, a.kappa, a.max_wind, a.workers, a.threads, a.time_limit)
    else:
        report(a.tag)

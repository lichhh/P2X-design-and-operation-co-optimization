"""
Scale-separated swap: does E5's hourly wind texture move the off-grid design?

Wind is split into a daily part d (24-h centred moving average) and an hourly residual
r = v - d. For each knob analog k01-k25 two hybrids keep the analog's daily part and PV:

    C_i  d(k_i) + r(k'_i)   residual of a second, independent knob analog k'_i with the
                            same knob sequence (control: real texture)
    X_i  d(k_i) + r(n_i)    residual of E5's synthetic year i (same knobs)

X - C is the effect of E5's hourly texture as a whole (|dv| quantiles, kurtosis, maximum
wind, sign_keep, |dv| clustering, sub-daily spectrum, between-window activity) with the
daily-scale weather held at real values. Both hybrids carry the same artefact of joining
a residual to another day's daily mean, so it cancels in X - C; C - k shows its size.
Annual wind CF of every hybrid is rescaled to that of k_i, as in perturbation_study.py.

A second test fixes the plant at the real-optimal design and dispatches it (perfect-foresight
LP, as in the test step) on C, X, k and n, so the same contrasts show the effect on operation.

Usage:
    python scale_swap_study.py prepare
    python scale_swap_study.py design --tag k125 --kappa 1.25
    python scale_swap_study.py dispatch --tag k125 --kappa 1.25
    python scale_swap_study.py report --tag k125
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from evaluate_analogs import load_history, stitch
from evaluate_ensemble import CAP_COLS, _pool, design_path, load_series
from evaluate_scenarios import OUT
from evaluate_synthetic_years import INPUTS as SYN_INPUTS
from evaluate_synthetic_years import REAL_CSV, knob_candidates, syn_path
from offgrid_design_gamspy import BARRIER, Capacities, OffgridCase, capacities_of, design_lp, solve_offgrid
from perturbation_study import calm_hours, daily_std, keep_cf
from run_ptm_examples import wind_power_cf

DATA = Path(r"C:\Users\qimen\Desktop\MILP scenario generation data\new 50 scenarios\csv")
INPUTS = OUT / "swap_inputs.npz"
PAIRS = [f"{i:02d}" for i in range(1, 26)]
HYBRIDS = {"C": "real daily + other real analog's hourly residual (control)", "X": "real daily + E5 hourly residual"}
WINDOW = 576


def split(v):
    d = pd.Series(v).rolling(24, center=True, min_periods=1).mean().to_numpy()
    return d, v - d


def texture(v) -> dict:
    """Hourly statistics the generator is tuned on, plus the two daily-scale ones the MILP is sensitive to."""

    dv = np.diff(v)
    a = np.abs(dv)
    c = a - a.mean()
    w = [dv[s : s + WINDOW].std() for s in range(0, len(dv) - WINDOW + 1, WINDOW)]
    q = np.quantile(a, [0.5, 0.75, 0.9, 0.99, 0.999])
    return {
        "|dv| p50": q[0], "|dv| p75": q[1], "|dv| p90": q[2], "|dv| p99": q[3], "|dv| p99.9": q[4],
        "kurt dv": stats.kurtosis(dv, fisher=False), "wind max": v.max(),
        "win_dvstd_cv": np.std(w) / np.mean(w),
        "sign_keep": np.mean(np.sign(dv[1:]) == np.sign(dv[:-1])),
        "acf |dv| lag 3": (c[:-3] * c[3:]).mean() / c.var(),
        "std daily mean v": daily_std(v), "calm h (CF<0.05)": calm_hours(v), "wind CF": wind_power_cf(v).mean(),
    }


def prepare() -> None:
    z = np.load(SYN_INPUTS)
    real = pd.read_csv(REAL_CSV, parse_dates=["time_utc"])
    knobs = pd.read_csv(DATA / "knobs_per_window.csv", index_col=0)
    wind_h, k_h, time_h, start = load_history(real)
    arrays, rows, overlap = {}, [], []
    for i in PAIRS:
        kn = knobs.loc[f"synthetic_year_{i}"].to_numpy(float)
        w2, _ = stitch(len(z[f"wind_k{i}"]), knob_candidates(kn, wind_h, time_h, start), wind_h, k_h, np.zeros(len(z[f"wind_k{i}"])), seed=9000 + int(i))
        vk, vn = z[f"wind_k{i}"].astype(float), z[f"wind_n{i}"].astype(float)
        overlap.append(np.mean(np.isclose(w2, vk, atol=1e-3)))
        dk, _ = split(vk)
        cf = z[f"cfw_k{i}"].mean()
        raw = {"C": np.maximum(dk + split(w2)[1], 0.0), "X": np.maximum(dk + split(vn)[1], 0.0)}
        for h, v in raw.items():
            s = keep_cf(v, cf)
            arrays[f"cfw_{h}{i}"] = wind_power_cf(s)
            arrays[f"cfp_{h}{i}"] = z[f"cfp_k{i}"]
            arrays[f"wind_{h}{i}"] = s.astype(np.float32)
            rows.append({"set": h, "pair": i, "CF scale": s.mean() / v.mean(), "clipped h": int((dk + (split(w2)[1] if h == "C" else split(vn)[1]) < 0).sum()), **texture(s)})
        rows.append({"set": "k (analog)", "pair": i, **texture(vk)})
        rows.append({"set": "n (E5)", "pair": i, **texture(vn)})
        rows.append({"set": "k' (2nd analog)", "pair": i, **texture(w2)})
    np.savez_compressed(INPUTS, **arrays)
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(f"second analog shares {100 * np.mean(overlap):.1f} % of hours with the first (same segment drawn twice)")
    print("\nmean over 25 pairs")
    print(t.groupby("set").mean(numeric_only=True).T.round(4).to_string())
    m = t.pivot_table(index="pair", columns="set")
    print("\nE5 texture shift: (X - C) versus (n - k), mean over pairs")
    for c in texture(np.arange(100.0)).keys():
        print(f"  {c:18s} X-C {(m[c]['X'] - m[c]['C']).mean():+9.4f}   n-k {(m[c]['n (E5)'] - m[c]['k (analog)']).mean():+9.4f}")


def load_swap(name: str):
    with np.load(INPUTS) as z:
        return z[f"cfw_{name}"], z[f"cfp_{name}"]


def design_task(args) -> dict:
    name, kappa, max_wind, threads, limit = args
    case = OffgridCase(kappa=kappa, max_wind_MW=max_wind)
    des = design_lp(*load_swap(name), case, time_limit=limit, threads=threads)
    ev = solve_offgrid(*load_series("real"), case, fixed=capacities_of(des), integer=False, time_limit=limit, threads=threads, solver_options=BARRIER)
    row = {"dataset": name}
    row.update({f"d_{k}": v for k, v in des.items()})
    row.update({f"e_{k}": v for k, v in ev.items() if k not in CAP_COLS + ["C_PEM_MW"]})
    return row


def result_path(tag: str) -> Path:
    return OUT / f"swap_design_{tag}.csv"


def design(tag, kappa, max_wind, workers, threads, limit) -> None:
    path = result_path(tag)
    done = set(pd.read_csv(path)["dataset"]) if path.exists() else set()
    names = [f"{h}{i}" for i in PAIRS for h in HYBRIDS if f"{h}{i}" not in done]
    print(f"{len(names)} hybrid designs to solve", flush=True)
    _pool(design_task, [(n, kappa, max_wind, threads, limit) for n in names], path, workers, lambda r: f"{r['dataset']} on real {r['e_TAC_usd_per_t']:.2f}")


def dispatch_path(tag: str) -> Path:
    return OUT / f"swap_dispatch_{tag}.csv"


def load_any(name: str):
    if name[0] in HYBRIDS:
        return load_swap(name)
    with np.load(SYN_INPUTS) as z:
        return z[f"cfw_{name}"], z[f"cfp_{name}"]


def dispatch_task(args) -> dict:
    name, caps, kappa, max_wind, threads, limit = args
    case = OffgridCase(kappa=kappa, max_wind_MW=max_wind)
    ev = solve_offgrid(*load_any(name), case, fixed=caps, integer=False, time_limit=limit, threads=threads, solver_options=BARRIER)
    return {"dataset": name, **{k: v for k, v in ev.items() if k not in CAP_COLS + ["C_PEM_MW"]}}


def dispatch(tag, kappa, max_wind, workers, threads, limit) -> None:
    r = pd.read_csv(design_path(tag)).set_index("dataset").loc["real"]
    caps = Capacities(r["d_C_wind_MW"], r["d_C_PV_MW"], r["d_N_PEM"], r["d_C_BESS_MWh"], r["d_C_H2_kg"])
    path = dispatch_path(tag)
    done = set(pd.read_csv(path)["dataset"]) if path.exists() else set()
    names = [f"{h}{i}" for i in PAIRS for h in ("C", "X", "k", "n") if f"{h}{i}" not in done]
    print(f"{len(names)} fixed-design dispatch solves, capacities {caps}", flush=True)
    _pool(dispatch_task, [(n, caps, kappa, max_wind, threads, limit) for n in names], path, workers, lambda r: f"{r['dataset']} {r['TAC_usd_per_t']:.2f}")


def paired(title, a, b):
    x = a - b
    se = x.std(ddof=1) / np.sqrt(len(x))
    tc = stats.t.ppf(0.975, len(x) - 1)
    print(f"  {title:24s} {x.mean():+10.3f} [{x.mean() - tc * se:+9.3f}, {x.mean() + tc * se:+9.3f}]  decrease {(x < 0).sum():2d}/{len(x)}")


def report_dispatch(tag: str) -> None:
    d = pd.read_csv(dispatch_path(tag)).set_index("dataset")
    print("\nFixed real-optimal plant, dispatch on each series. status:", d["status"].value_counts().to_dict(), f"n={len(d)}")
    cols = {
        "cost USD/t": "TAC_usd_per_t", "penalty USD/t": "penalty_usd_per_t", "shortfall t/y": "shortfall_t_y",
        "unserved MWh/y": "unserved_MWh_y", "battery out MWh/y": "BESS_MWh_y", "H2 tank out t/y": "H2_tank_t_y",
        "PEM starts/y": "PEM_starts_y", "PEM standby h/y": "PEM_standby_module_h_y", "curtail share": "curtail_share",
        "MeOH load std": "MeOH_load_std",
    }
    for title, pa, pb in (("X - C: E5 hourly texture", "X", "C"), ("C - k: swap artefact", "C", "k"), ("n - k: E5 total (for scale)", "n", "k")):
        idx = [i for i in PAIRS if f"{pa}{i}" in d.index and f"{pb}{i}" in d.index]
        if len(idx) < 2:
            continue
        print(f"\n{title} (n={len(idx)})")
        for lab, c in cols.items():
            paired(lab, d.loc[[f"{pa}{i}" for i in idx], c].to_numpy(float), d.loc[[f"{pb}{i}" for i in idx], c].to_numpy(float))


def report(tag: str) -> None:
    best = pd.read_csv(design_path(tag)).set_index("dataset").loc["real", "e_TAC_usd_per_t"]
    if not result_path(tag).exists():
        report_dispatch(tag)
        return
    d = pd.read_csv(result_path(tag)).set_index("dataset")
    base = pd.read_csv(syn_path(tag)).set_index("dataset") if syn_path(tag).exists() else None
    print("status:", d["d_status"].value_counts().to_dict(), d["e_status"].value_counts().to_dict(), f"n={len(d)}")
    cols = {
        "own cost USD/t": "d_TAC_usd_per_t", "H2 tank kg": "d_C_H2_kg", "wind MW": "d_C_wind_MW", "PV MW": "d_C_PV_MW",
        "battery MWh": "d_C_BESS_MWh", "PEM MW": "d_C_PEM_MW", "regret on real %": "e_TAC_usd_per_t", "shortfall on real t/y": "e_shortfall_t_y",
    }

    def get(frame, names, c):
        x = frame.loc[names, c].to_numpy(float)
        return 100 * (x / best - 1) if c == "e_TAC_usd_per_t" else x

    contrasts = [("X - C: E5 hourly texture", d, "X", d, "C")]
    if base is not None:
        contrasts += [("C - k: swap artefact", d, "C", base, "k"), ("n - k: E5 total (for scale)", base, "n", base, "k")]
    print("\nPaired effect: mean [95% CI] | pairs with decrease")
    for title, fa, pa, fb, pb in contrasts:
        idx = [i for i in PAIRS if f"{pa}{i}" in fa.index and f"{pb}{i}" in fb.index]
        print(f"\n{title} (n={len(idx)})")
        for lab, c in cols.items():
            x = get(fa, [f"{pa}{i}" for i in idx], c) - get(fb, [f"{pb}{i}" for i in idx], c)
            se = x.std(ddof=1) / np.sqrt(len(x))
            tc = stats.t.ppf(0.975, len(x) - 1)
            print(f"  {lab:22s} {x.mean():+10.2f} [{x.mean() - tc * se:+9.2f}, {x.mean() + tc * se:+9.2f}]  decrease {(x < 0).sum():2d}/{len(x)}")
    if dispatch_path(tag).exists():
        report_dispatch(tag)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["prepare", "design", "dispatch", "report"])
    ap.add_argument("--tag", default="k125")
    ap.add_argument("--kappa", type=float, default=1.25)
    ap.add_argument("--max-wind", type=float, default=500.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--time-limit", type=float, default=3600.0)
    o = ap.parse_args()
    if o.command == "prepare":
        prepare()
    elif o.command == "design":
        design(o.tag, o.kappa, o.max_wind, o.workers, o.threads, o.time_limit)
    elif o.command == "dispatch":
        dispatch(o.tag, o.kappa, o.max_wind, o.workers, o.threads, o.time_limit)
    else:
        report(o.tag)

"""
Single-metric perturbation experiments: which weather statistics move the off-grid design?

Base series are knob-matched analog years k01-k25 (real weather, 2023 calendar) from
evaluate_synthetic_years.py, whose designs are already solved. Each perturbation
changes one statistic by the bias found in E5 (or by one standard deviation of real
years where E5's bias is too small to detect), keeps the annual wind capacity factor
by a small rescaling of wind speed, and is then designed and tested on real 2024-2025
like the base series. Paired differences to the base designs are causal effects.

    P0  none (repeat solve of k01-k10, numerical noise of the pipeline)
    P1  day-to-day wind variability: 24-h mean anomalies about the 24-day mean scaled
        so the std of daily mean wind drops as in E5 (-3.9 %)
    P2  calm hours: wind below the CF 0.05 speed (4.47 m/s) compressed towards a
        threshold so hours with CF < 0.05 drop as in E5 (-8 %)
    P3  wind-PV complementarity: PV days reordered within 12-day blocks so the daily
        wind-PV correlation falls by one real-year std (E5 bias is -0.28 std)
    P4  diurnal wind cycle (expected negative control): hour-of-day wind anomaly
        amplified so the diurnal range of wind CF rises as in E5 (+20 %)
    P5  PV level -3 % (E5 versus real 2024-2025; versus analogs E5 has no bias)
    P6  day-to-day PV variability: daily PV anomalies about the 24-day mean scaled so
        the std of daily PV CF falls by one real-year std, annual PV CF kept

For configurations other than the base case (e.g. PV-heavy), the 25 base designs are
solved here as P0 and serve as the reference.

Usage:
    python perturbation_study.py prepare
    python perturbation_study.py design --tag k125 --kappa 1.25
    python perturbation_study.py design --tag pvheavy --kappa 1.25 --max-wind 10
    python perturbation_study.py report --tag pvheavy
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import stats

from evaluate_ensemble import CAP_COLS, _pool, design_path, load_series
from evaluate_scenarios import OUT, SEGMENT_H
from evaluate_synthetic_years import INPUTS as SYN_INPUTS
from evaluate_synthetic_years import syn_path
from offgrid_design_gamspy import BARRIER, OffgridCase, capacities_of, design_lp, solve_offgrid
from run_ptm_examples import wind_power_cf

INPUTS = OUT / "perturb_inputs.npz"
BASE = [f"k{i:02d}" for i in range(1, 26)]
NULL = BASE[:10]
PERTURBATIONS = ["P0", "P1", "P2", "P3", "P4", "P5", "P6"]
LABEL = {
    "P0": "none (repeat solve)",
    "P1": "day-to-day wind variability -3.9 %",
    "P2": "calm hours -8 %",
    "P3": "wind-PV daily correlation -1 std",
    "P4": "diurnal wind range +20 %",
    "P5": "PV level -3 %",
    "P6": "day-to-day PV variability -1 std",
}
PV_LEVEL = 0.97
V_CALM = (1.5**3 + 0.05 * (12.0**3 - 1.5**3)) ** (1 / 3)  # wind speed of CF 0.05
TIMES = pd.date_range("2023-01-01", periods=8760, freq="h")
HOUR, MONTH = TIMES.hour.to_numpy(), TIMES.month.to_numpy()


# ---------------------------------------------------------------- statistics
def daily_std(v):
    return v[: len(v) // 24 * 24].reshape(-1, 24).mean(1).std()


def calm_hours(v):
    return float((wind_power_cf(v) < 0.05).sum())


def diurnal_range(v):
    p = pd.Series(wind_power_cf(v)).groupby(HOUR).mean()
    return p.max() - p.min()


def daily_corr(v, cfp):
    return np.corrcoef(wind_power_cf(v).reshape(-1, 24).mean(1), cfp.reshape(-1, 24).mean(1))[0, 1]


def pv_daily(cfp):
    return cfp[: len(cfp) // 24 * 24].reshape(-1, 24).mean(1)


def pv_daily_std(cfp):
    return pv_daily(cfp).std()


def pv_lag1(cfp):
    d = pv_daily(cfp)
    return np.corrcoef(d[:-1], d[1:])[0, 1]


def pv_dark_run(cfp):
    """Longest run of days below half the 31-day mean PV CF."""

    d = pd.Series(pv_daily(cfp))
    dark = (d < 0.5 * d.rolling(31, center=True, min_periods=1).mean()).astype(int)
    return float(dark.groupby((dark.diff() != 0).cumsum()).sum().max())


# ---------------------------------------------------------------- perturbations
def keep_cf(v_new, cf_target):
    """Rescale wind speed so the annual wind CF equals cf_target."""

    lo, hi = 0.8, 1.25
    for _ in range(40):
        s = 0.5 * (lo + hi)
        lo, hi = (s, hi) if wind_power_cf(s * v_new).mean() < cf_target else (lo, s)
    return 0.5 * (lo + hi) * v_new


def p1(v, f):
    d = pd.Series(v).rolling(24, center=True, min_periods=1).mean().to_numpy()
    m = pd.Series(v).rolling(576, center=True, min_periods=1).mean().to_numpy()
    return np.maximum(v + (f - 1.0) * (d - m), 0.0)


def p2(v, g):
    u = V_CALM + 1.0
    return np.where(v < u, u - (u - v) * g, v)


def p4(v, f):
    d = pd.Series(v).rolling(24, center=True, min_periods=1).mean().to_numpy()
    prof = pd.Series(v - d).groupby([MONTH, HOUR]).transform("mean").to_numpy()
    return np.maximum(v + (f - 1.0) * prof, 0.0)


def p3(v, cfp, lam):
    """Reorder whole PV days within 12-day blocks towards anti-correlation with wind."""

    w_day = wind_power_cf(v).reshape(-1, 24).mean(1)
    days = cfp.reshape(-1, 24)
    p_day = days.mean(1)
    order = np.arange(len(days))
    for s in range(0, len(days), SEGMENT_H // 24):
        idx = np.arange(s, min(s + SEGMENT_H // 24, len(days)))
        if len(idx) < 3:
            continue
        zw = (w_day[idx] - w_day[idx].mean()) / (w_day[idx].std() + 1e-9)
        zp = (p_day[idx] - p_day[idx].mean()) / (p_day[idx].std() + 1e-9)
        key = lam * (-zw) + (1.0 - lam) * zp
        order[idx[np.argsort(key)]] = idx[np.argsort(p_day[idx])]
    return days[order].reshape(-1)


def keep_pv(c, target):
    """Rescale PV CF (clipped at 1) so its annual mean equals target."""

    lo, hi = 0.8, 1.25
    for _ in range(40):
        s = 0.5 * (lo + hi)
        lo, hi = (s, hi) if np.minimum(s * c, 1.0).mean() < target else (lo, s)
    return np.minimum(0.5 * (lo + hi) * c, 1.0)


def p6(cfp, f):
    days = cfp.reshape(-1, 24)
    d = days.mean(1)
    m = pd.Series(d).rolling(24, center=True, min_periods=1).mean().to_numpy()
    new = np.maximum(m + f * (d - m), 0.0)
    r = np.divide(new, d, out=np.ones_like(d), where=d > 1e-6)
    return keep_pv(np.clip(days * r[:, None], 0.0, 1.0).reshape(-1), cfp.mean())


def calibrate(apply, measure, base_values, target, lo, hi, increasing):
    """Bisection on one strength parameter so the mean statistic over BASE hits target."""

    for _ in range(30):
        x = 0.5 * (lo + hi)
        val = np.mean([measure(apply(b, x)) for b in base_values])
        if (val < target) == increasing:
            lo = x
        else:
            hi = x
    return 0.5 * (lo + hi)


def prepare() -> None:
    z = np.load(SYN_INPUTS)
    wind = {n: z[f"wind_{n}"].astype(float) for n in BASE}
    cfp = {n: z[f"cfp_{n}"] for n in BASE}
    cfw = {n: z[f"cfw_{n}"].mean() for n in BASE}
    syn = [f"n{i:02d}" for i in range(1, 51)]
    ana = [f"k{i:02d}" for i in range(1, 51)]

    def ratio(fun):
        return np.mean([fun(z[f"wind_{n}"].astype(float)) for n in syn]) / np.mean([fun(z[f"wind_{n}"].astype(float)) for n in ana])

    r_daily, r_calm, r_diurnal = ratio(daily_std), ratio(calm_hours), ratio(diurnal_range)
    real_years = ana + [f"y{y}" for y in range(2005, 2024) if y != 2021]
    corr_sd = np.std([np.corrcoef(z[f"cfw_{n}"][:8760].reshape(-1, 24).mean(1), z[f"cfp_{n}"][:8760].reshape(-1, 24).mean(1))[0, 1] for n in real_years], ddof=1)
    print(f"E5 / analog ratios: daily std {r_daily:.4f}, calm hours {r_calm:.4f}, diurnal range {r_diurnal:.4f}; real-year std of daily wind-PV corr {corr_sd:.4f}")

    items = [(n, (wind[n], cfw[n])) for n in BASE]
    wrap = lambda fun: (lambda b, x: keep_cf(fun(b[0], x), b[1]))  # noqa: E731
    base = [b for _, b in items]
    f1 = calibrate(wrap(p1), daily_std, base, r_daily * np.mean([daily_std(b[0]) for b in base]), 0.5, 1.0, increasing=True)
    g2 = calibrate(wrap(p2), calm_hours, base, r_calm * np.mean([calm_hours(b[0]) for b in base]), 0.0, 1.0, increasing=True)
    f4 = calibrate(wrap(p4), diurnal_range, base, r_diurnal * np.mean([diurnal_range(b[0]) for b in base]), 1.0, 3.0, increasing=True)
    c0 = np.mean([daily_corr(wind[n], cfp[n]) for n in BASE])
    lo, hi = 0.0, 1.0
    for _ in range(30):
        lam = 0.5 * (lo + hi)
        c = np.mean([daily_corr(wind[n], p3(wind[n], cfp[n], lam)) for n in BASE])
        lo, hi = (lam, hi) if c > c0 - corr_sd else (lo, lam)

    print("\nPV statistics: E5 mean | analog mean | E5 bias in real-year std")
    for lab, fun in (("PV CF", np.mean), ("std daily PV CF", pv_daily_std), ("lag-1 day autocorr PV", pv_lag1), ("longest dark-day run", pv_dark_run)):
        a = np.array([fun(z[f"cfp_{n}"]) for n in syn])
        b = np.array([fun(z[f"cfp_{n}"]) for n in ana])
        sd = np.std([fun(z[f"cfp_{n}"][:8760]) for n in real_years], ddof=1)
        print(f"  {lab:22s} {a.mean():8.4f} | {b.mean():8.4f} | {(a.mean() - b.mean()) / sd:+.2f}  (Wilcoxon p {stats.wilcoxon(a, b).pvalue:.3f})")
    pv_sd = np.std([pv_daily_std(z[f"cfp_{n}"][:8760]) for n in real_years], ddof=1)
    f6 = calibrate(lambda c, x: p6(c, x), pv_daily_std, [cfp[n] for n in BASE], np.mean([pv_daily_std(cfp[n]) for n in BASE]) - pv_sd, 0.0, 1.0, increasing=True)
    print(f"strengths: P1 f {f1:.3f}, P2 g {g2:.3f}, P3 lambda {lam:.3f}, P4 f {f4:.3f}, P6 f {f6:.3f}")

    arrays = {}
    for n in BASE:
        v, cp = wind[n], cfp[n]
        series = {
            "P0": (v, cp),
            "P1": (keep_cf(p1(v, f1), cfw[n]), cp),
            "P2": (keep_cf(p2(v, g2), cfw[n]), cp),
            "P3": (v, p3(v, cp, lam)),
            "P4": (keep_cf(p4(v, f4), cfw[n]), cp),
            "P5": (v, PV_LEVEL * cp),
            "P6": (v, p6(cp, f6)),
        }
        for p, (w, c) in series.items():
            arrays[f"cfw_{p}_{n}"] = wind_power_cf(w)
            arrays[f"cfp_{p}_{n}"] = c
            arrays[f"wind_{p}_{n}"] = w.astype(np.float32)
    np.savez_compressed(INPUTS, **arrays)

    print("\nachieved change over k01-k25 (perturbed / base), annual wind CF and PV CF kept")
    rows = []
    for p in PERTURBATIONS[1:]:
        w = [arrays[f"wind_{p}_{n}"].astype(float) for n in BASE]
        rows.append({
            "perturbation": LABEL[p],
            "daily std": np.mean([daily_std(x) for x in w]) / np.mean([daily_std(wind[n]) for n in BASE]),
            "calm hours": np.mean([calm_hours(x) for x in w]) / np.mean([calm_hours(wind[n]) for n in BASE]),
            "diurnal range": np.mean([diurnal_range(x) for x in w]) / np.mean([diurnal_range(wind[n]) for n in BASE]),
            "hourly std v": np.mean([x.std() for x in w]) / np.mean([wind[n].std() for n in BASE]),
            "mean v": np.mean([x.mean() for x in w]) / np.mean([wind[n].mean() for n in BASE]),
            "wind CF": np.mean([arrays[f"cfw_{p}_{n}"].mean() for n in BASE]) / np.mean(list(cfw.values())),
            "PV CF": np.mean([arrays[f"cfp_{p}_{n}"].mean() for n in BASE]) / np.mean([cfp[n].mean() for n in BASE]),
            "d corr wind-PV": np.mean([daily_corr(wind[n], arrays[f"cfp_{p}_{n}"]) - daily_corr(wind[n], cfp[n]) for n in BASE]),
            "std daily PV": np.mean([pv_daily_std(arrays[f"cfp_{p}_{n}"]) for n in BASE]) / np.mean([pv_daily_std(cfp[n]) for n in BASE]),
            "PV lag-1": np.mean([pv_lag1(arrays[f"cfp_{p}_{n}"]) - pv_lag1(cfp[n]) for n in BASE]),
        })
    print(pd.DataFrame(rows).set_index("perturbation").round(4).to_string())


# ---------------------------------------------------------------- solves
def load_perturbed(name: str):
    with np.load(INPUTS) as z:
        return z[f"cfw_{name}"], z[f"cfp_{name}"]


def design_task(args) -> dict:
    name, kappa, max_wind, threads, limit = args
    case = OffgridCase(kappa=kappa, max_wind_MW=max_wind)
    des = design_lp(*load_perturbed(name), case, time_limit=limit, threads=threads)
    ev = solve_offgrid(*load_series("real"), case, fixed=capacities_of(des), integer=False, time_limit=limit, threads=threads, solver_options=BARRIER)
    row = {"dataset": name}
    row.update({f"d_{k}": v for k, v in des.items()})
    row.update({f"e_{k}": v for k, v in ev.items() if k not in CAP_COLS + ["C_PEM_MW"]})
    return row


def result_path(tag: str):
    return OUT / f"perturb_design_{tag}.csv"


def has_base(tag: str) -> bool:
    """Base designs of k01-k25 already solved by evaluate_synthetic_years.py for this tag."""

    return syn_path(tag).exists()


def design(tag, kappa, max_wind, workers, threads, limit) -> None:
    path = result_path(tag)
    done = set(pd.read_csv(path)["dataset"]) if path.exists() else set()
    p0 = NULL if has_base(tag) else BASE
    names = [f"P0_{n}" for n in p0] + [f"{p}_{n}" for n in BASE for p in PERTURBATIONS[1:]]
    names = [n for n in names if n not in done]
    print(f"{len(names)} perturbed designs to solve", flush=True)
    _pool(design_task, [(n, kappa, max_wind, threads, limit) for n in names], path, workers, lambda r: f"{r['dataset']} on real {r['e_TAC_usd_per_t']:.2f}")


def report(tag: str) -> None:
    best = pd.read_csv(design_path(tag)).set_index("dataset").loc["real", "e_TAC_usd_per_t"]
    d = pd.read_csv(result_path(tag)).set_index("dataset")
    if has_base(tag):
        base = pd.read_csv(syn_path(tag)).set_index("dataset")
    else:
        base = d.loc[[f"P0_{n}" for n in BASE if f"P0_{n}" in d.index]].rename(index=lambda s: s[3:])
        print("base designs: P0 solved here, n =", len(base))
    print("status:", d["d_status"].value_counts().to_dict(), d["e_status"].value_counts().to_dict(), f"n={len(d)}")
    cols = {
        "own cost USD/t": "d_TAC_usd_per_t", "H2 tank kg": "d_C_H2_kg", "wind MW": "d_C_wind_MW", "PV MW": "d_C_PV_MW",
        "battery MWh": "d_C_BESS_MWh", "PEM MW": "d_C_PEM_MW", "regret on real %": "e_TAC_usd_per_t", "shortfall on real t/y": "e_shortfall_t_y",
    }
    print("\nPaired effect (perturbed - base): mean [95% CI] | pairs with decrease")
    for p in PERTURBATIONS:
        names = [n for n in BASE if f"{p}_{n}" in d.index and n in base.index]
        if not names or (p == "P0" and not has_base(tag)):
            continue
        print(f"\n{p} {LABEL[p]} (n={len(names)})")
        for lab, c in cols.items():
            a = d.loc[[f"{p}_{n}" for n in names], c].to_numpy(float)
            b = base.loc[names, c].to_numpy(float)
            if c == "e_TAC_usd_per_t":
                a, b = 100 * (a / best - 1), 100 * (b / best - 1)
            x = a - b
            se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
            tcrit = stats.t.ppf(0.975, len(x) - 1) if len(x) > 1 else np.nan
            print(f"  {lab:22s} {x.mean():+10.2f} [{x.mean() - tcrit * se:+9.2f}, {x.mean() + tcrit * se:+9.2f}]  decrease {(x < 0).sum():2d}/{len(x)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["prepare", "design", "report"])
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
    else:
        report(o.tag)

"""
Independent real-weather benchmark for the scenario ensemble (off-grid PtM).

Each analog is built like a generated scenario, but from observed weather outside the
test period. For every 12-day segment of 2024-2025, a 24-day window is drawn from
2005-2023 (without 2021), within 30 days of the same season, among the windows whose
mean wind is closest to the real 24-day mean. The centre 12 days are kept and joined
with a 6-h crossfade. Solar uses the clear-sky index of the source window.

The analogs in evaluate_scenarios.py resample 2024 and 2025 from each other, so they
share weather with the test series. These analogs do not.

Usage:
    python evaluate_analogs.py prepare --data <folder with real_2024_2025.csv and scenario_XX.csv>
    python evaluate_analogs.py design --tag k125 --kappa 1.25
    python evaluate_analogs.py report --tag k125
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from evaluate_ensemble import (
    BLOCKS,
    CAP_COLS,
    HISTORY,
    SCENARIOS,
    SUBSET_REPEATS,
    SUBSET_SIZES,
    _pool,
    design_path,
    load_series,
    mean_caps,
    power_year,
    quant,
)
from evaluate_scenarios import CROSSFADE_H, OUT, PV_PR, SEGMENT_H, WINDOW_H, clear_sky_index
from offgrid_design_gamspy import BARRIER, Capacities, OffgridCase, capacities_of, design_lp, solve_offgrid
from run_ptm_examples import wind_power_cf

INPUTS = OUT / "analog_inputs.npz"
N_ANALOG = 50
ANALOGS = [f"a{i:02d}" for i in range(1, N_ANALOG + 1)]
YEARS = [y for block in BLOCKS for y in block]
SEASON_DAYS = 30
TOP_MATCHES = 20


def hours(index) -> np.ndarray:
    return ((pd.DatetimeIndex(index) - pd.Timestamp("2000-01-01")) / pd.Timedelta(hours=1)).to_numpy().round().astype(np.int64)


def load_history(real: pd.DataFrame):
    frames = [power_year(y) for y in YEARS]
    start, pos = {}, 0
    for y, f in zip(YEARS, frames):
        start[y] = pos
        pos += len(f)
    h = pd.concat(frames)
    clim = real.groupby([real["time_utc"].dt.dayofyear, real["time_utc"].dt.hour])["ghi_clearsky_wm2"].mean()
    cs = clim.reindex(pd.MultiIndex.from_arrays([h.index.dayofyear, h.index.hour])).to_numpy(float)
    k = clear_sky_index(h["ghi_wm2"].to_numpy(float), np.nan_to_num(cs))
    return h["wind_ms"].to_numpy(float), k, hours(h.index), start


def segment_candidates(real: pd.DataFrame, wind_h, time_h, start) -> dict:
    wind = real["wind_ms"].to_numpy(float)
    n = len(wind)
    real_cum = np.concatenate([[0.0], np.cumsum(wind)])
    hist_cum = np.concatenate([[0.0], np.cumsum(wind_h)])
    times = real["time_utc"]
    t_real = hours(times)
    out = {}
    for seg in range(0, n, SEGMENT_H):
        ctx = int(np.clip(seg + SEGMENT_H // 2 - WINDOW_H // 2, 0, n - WINDOW_H))
        target = (real_cum[ctx + WINDOW_H] - real_cum[ctx]) / WINDOW_H
        offset = int((times.iloc[ctx] - pd.Timestamp(times.iloc[ctx].year, 1, 1)) / pd.Timedelta(hours=1))
        a = np.array([start[y] + offset + 24 * j for y in YEARS for j in range(-SEASON_DAYS, SEASON_DAYS + 1)])
        a = a[(a >= 0) & (a + WINDOW_H <= len(wind_h))]
        a = a[(time_h[a + WINDOW_H - 1] - time_h[a] == WINDOW_H - 1) & ((time_h[a] - t_real[ctx]) % 24 == 0)]
        mean = (hist_cum[a + WINDOW_H] - hist_cum[a]) / WINDOW_H
        out[seg] = (ctx, a[np.argsort(np.abs(mean - target))[:TOP_MATCHES]])
    return out


def stitch(n, candidates, wind_h, k_h, clear_sky, seed: int):
    rng = np.random.default_rng(seed)
    w_sum, k_sum, weight = np.zeros(n), np.zeros(n), np.zeros(n)
    half = CROSSFADE_H // 2
    for seg, (ctx, cand) in candidates.items():
        a = int(cand[rng.integers(len(cand))])
        idx = np.arange(max(0, seg - half), min(n, seg + SEGMENT_H + half))
        ramp = np.minimum.reduce(
            [np.ones(len(idx)), (idx - (seg - half) + 0.5) / CROSSFADE_H, (seg + SEGMENT_H + half - idx - 0.5) / CROSSFADE_H]
        )
        src = np.clip(a + idx - ctx, 0, len(wind_h) - 1)
        w_sum[idx] += ramp * wind_h[src]
        k_sum[idx] += ramp * k_h[src]
        weight[idx] += ramp
    return w_sum / weight, (k_sum / weight) * clear_sky


def window_error(wind, ref) -> float:
    """RMS error of the 24-day mean wind against the real series, m/s."""

    a = pd.Series(wind).rolling(WINDOW_H).mean().to_numpy()[WINDOW_H - 1 :: SEGMENT_H]
    b = pd.Series(ref).rolling(WINDOW_H).mean().to_numpy()[WINDOW_H - 1 :: SEGMENT_H]
    return float(np.sqrt(np.mean((a - b) ** 2)))


def prepare(data_dir: str) -> None:
    d = Path(data_dir)
    real = pd.read_csv(d / "real_2024_2025.csv", parse_dates=["time_utc"])
    w_real = real["wind_ms"].to_numpy(float)
    wind_h, k_h, time_h, start = load_history(real)
    cands = segment_candidates(real, wind_h, time_h, start)
    arrays, wind_all = {}, {}
    for i, name in enumerate(ANALOGS, 1):
        w, g = stitch(len(real), cands, wind_h, k_h, real["ghi_clearsky_wm2"].to_numpy(float), seed=5000 + i)
        wind_all[name] = w
        arrays[f"cfw_{name}"] = wind_power_cf(w)
        arrays[f"cfp_{name}"] = np.clip(PV_PR * g / 1000.0, 0.0, 1.0)
    np.savez_compressed(INPUTS, **arrays)
    gen = {n: pd.read_csv(d / f"scenario_{i:02d}.csv")["wind_ms"].to_numpy(float) for i, n in enumerate(SCENARIOS, 1)}
    print("real      mean wind %.3f m/s" % w_real.mean())
    for label, group in (("scenarios", gen), ("analogs", wind_all)):
        err = [window_error(x, w_real) for x in group.values()]
        print(f"{label:9s} mean wind {np.mean([x.mean() for x in group.values()]):.3f} (sd {np.std([x.mean() for x in group.values()], ddof=1):.3f}) | 24-day mean wind RMS error vs real {np.mean(err):.3f} m/s")


def load_analog(name: str):
    with np.load(INPUTS) as z:
        return z[f"cfw_{name}"], z[f"cfp_{name}"]


def design_task(args) -> dict:
    name, kappa, max_wind, threads, limit = args
    case = OffgridCase(kappa=kappa, max_wind_MW=max_wind)
    des = design_lp(*load_analog(name), case, time_limit=limit, threads=threads)
    ev = solve_offgrid(*load_series("real"), case, fixed=capacities_of(des), integer=False, time_limit=limit, threads=threads, solver_options=BARRIER)
    row = {"dataset": name}
    row.update({f"d_{k}": v for k, v in des.items()})
    row.update({f"e_{k}": v for k, v in ev.items() if k not in CAP_COLS + ["C_PEM_MW"]})
    return row


def consensus_task(args) -> dict:
    key, caps, kappa, max_wind, threads, limit = args
    case = OffgridCase(kappa=kappa, max_wind_MW=max_wind)
    ev = solve_offgrid(*load_series("real"), case, fixed=Capacities(**caps), integer=False, time_limit=limit, threads=threads, solver_options=BARRIER)
    keep = ("TAC_usd_per_t", "fixed_usd_per_t", "penalty_usd_per_t", "shortfall_t_y", "status", "solve_s")
    return {**key, **caps, **{k: ev[k] for k in keep}}


def analog_path(tag: str) -> Path:
    return OUT / f"analog_design_{tag}.csv"


def analog_consensus_path(tag: str) -> Path:
    return OUT / f"analog_consensus_{tag}.csv"


def design(tag, kappa, max_wind, workers, threads, limit) -> None:
    path = analog_path(tag)
    done = set(pd.read_csv(path)["dataset"]) if path.exists() else set()
    names = [n for n in ANALOGS if n not in done]
    print(f"{len(names)} analog designs to solve", flush=True)
    _pool(design_task, [(n, kappa, max_wind, threads, limit) for n in names], path, workers, lambda r: f"{r['dataset']} on real {r['e_TAC_usd_per_t']:.2f}")

    a = pd.read_csv(path).set_index("dataset").loc[ANALOGS]
    rng = np.random.default_rng(2026)
    keys = [({"K": k, "rep": r}, mean_caps(a.loc[rng.choice(ANALOGS, k, replace=False)])) for k in SUBSET_SIZES for r in range(SUBSET_REPEATS)]
    keys.append(({"K": N_ANALOG, "rep": 0}, mean_caps(a)))
    out = analog_consensus_path(tag)
    seen = set()
    if out.exists():
        old = pd.read_csv(out)
        seen = set(zip(old["K"], old["rep"]))
    tasks = [(key, caps, kappa, max_wind, threads, limit) for key, caps in keys if (key["K"], key["rep"]) not in seen]
    print(f"{len(tasks)} consensus tests to solve", flush=True)
    _pool(consensus_task, tasks, out, workers, lambda r: f"K={r['K']} on real {r['TAC_usd_per_t']:.2f}")


def report(tag: str) -> None:
    d = pd.read_csv(design_path(tag)).set_index("dataset")
    a = pd.read_csv(analog_path(tag)).set_index("dataset")
    print("analog status:", a["d_status"].value_counts().to_dict(), a["e_status"].value_counts().to_dict(), f"n={len(a)}")
    best = d.loc["real", "e_TAC_usd_per_t"]
    pct = lambda s: 100.0 * (np.asarray(s, dtype=float) / best - 1.0)  # noqa: E731
    s = d.loc[SCENARIOS]
    h = d.loc[[n for n in HISTORY if n in d.index]]
    rs, ra, rh = pct(s["e_TAC_usd_per_t"]), pct(a["e_TAC_usd_per_t"]), pct(h["e_TAC_usd_per_t"])
    print(f"\nReal-optimal cost {best:.2f} USD/t; regret in % of it")
    print("  50 scenarios       :", quant(rs))
    print(f"  {len(a)} indep. analogs  :", quant(ra))
    print(f"  {len(h)} history blocks   :", quant(rh))
    rng = np.random.default_rng(0)
    diff = rng.choice(rs, (20000, len(rs))).mean(1) - rng.choice(ra, (20000, len(ra))).mean(1)
    print("  scenarios - analogs, mean regret: %+.2f points, bootstrap 95%% CI %+.2f to %+.2f; Mann-Whitney p = %.3f" % (rs.mean() - ra.mean(), *np.percentile(diff, [2.5, 97.5]), stats.mannwhitneyu(rs, ra).pvalue))
    print("  shortfall > 1 t/y on real: scenarios %d/%d, analogs %d/%d, history %d/%d" % ((s.e_shortfall_t_y > 1).sum(), len(s), (a.e_shortfall_t_y > 1).sum(), len(a), (h.e_shortfall_t_y > 1).sum(), len(h)))

    print("\nDesigns: real | scenarios mean (sd) | analogs mean (sd)")
    for c in ["C_wind_MW", "C_PV_MW", "C_PEM_MW", "C_BESS_MWh", "C_H2_kg", "TAC_usd_per_t"]:
        x, y = s[f"d_{c}"], a[f"d_{c}"]
        print(f"  {c:15s} {d.loc['real', f'd_{c}']:9.2f} | {x.mean():9.2f} ({x.std(ddof=1):7.2f}) | {y.mean():9.2f} ({y.std(ddof=1):7.2f})")

    with np.load(OUT / "ensemble_inputs.npz") as z:
        cf_s = np.array([[z[f"cfw_{n}"].mean(), z[f"cfp_{n}"].mean()] for n in SCENARIOS])
        cf_r = np.array([z["cfw_real"].mean(), z["cfp_real"].mean()])
    with np.load(INPUTS) as z:
        cf_a = np.array([[z[f"cfw_{n}"].mean(), z[f"cfp_{n}"].mean()] for n in a.index])
    print("\nCapacity factors: real | scenarios mean (sd) | analogs mean (sd)")
    for j, lab in enumerate(("wind", "PV")):
        print(f"  {lab:4s} {cf_r[j]:.4f} | {cf_s[:, j].mean():.4f} ({cf_s[:, j].std(ddof=1):.4f}) | {cf_a[:, j].mean():.4f} ({cf_a[:, j].std(ddof=1):.4f})")

    c = pd.read_csv(analog_consensus_path(tag))
    g = pd.read_csv(OUT / f"ensemble_consensus_{tag}.csv")
    g = g[g["kind"] == "subset"]
    print("\nConsensus regret on real, %: K | scenarios mean | analogs mean")
    print(f"  {1:2d} | {rs.mean():6.3f} | {ra.mean():6.3f}")
    for k in [*SUBSET_SIZES, N_ANALOG]:
        print(f"  {k:2d} | {pct(g.loc[g['K'] == k, 'TAC_usd_per_t']).mean():6.3f} | {pct(c.loc[c['K'] == k, 'TAC_usd_per_t']).mean():6.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["prepare", "design", "report"])
    ap.add_argument("--data")
    ap.add_argument("--tag", default="k125")
    ap.add_argument("--kappa", type=float, default=1.25)
    ap.add_argument("--max-wind", type=float, default=500.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--time-limit", type=float, default=3600.0)
    o = ap.parse_args()
    if o.command == "prepare":
        prepare(o.data)
    elif o.command == "design":
        design(o.tag, o.kappa, o.max_wind, o.workers, o.threads, o.time_limit)
    else:
        report(o.tag)

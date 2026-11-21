"""
Decision-based evaluation of synthetic years (E5 with block-bootstrap knobs, off-grid PtM).

Every one-year series designs the plant, and the design is operated on the real
2024-2025 series. Regret is relative to the design made on the real two years.

Series:
    n01-n50    synthetic years (2023 calendar), knobs bootstrapped from 2024-2025
    k01-k50    knob-matched analogs: the same 32 knobs per year, but every 12-day
               segment is real weather from 2005-2023 (without 2021) whose 24-day
               mean wind matches the knob, stitched like the generator. Pair i
               shares the knobs, so n_i - k_i isolates what E5 adds given the knobs.
    yYYYY      single observed years 2005-2023 (without 2021), no knowledge of 2024-2025
    r2024/5    single real years of the test period (in-sample, shows the cost of a
               one-year design basis)

Usage:
    python evaluate_synthetic_years.py prepare --data <folder with synthetic_year_XX.csv and knobs_per_window.csv>
    python evaluate_synthetic_years.py design --tag k125 --kappa 1.25
    python evaluate_synthetic_years.py report --tag k125
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from evaluate_analogs import TOP_MATCHES, YEARS, load_history, stitch
from evaluate_ensemble import CAP_COLS, SUBSET_REPEATS, SUBSET_SIZES, _pool, design_path, load_series, mean_caps, power_year, quant
from evaluate_scenarios import OUT, PV_PR, SEGMENT_H, WINDOW_H
from offgrid_design_gamspy import BARRIER, Capacities, OffgridCase, capacities_of, design_lp, solve_offgrid
from run_ptm_examples import wind_power_cf

INPUTS = OUT / "synthetic_inputs.npz"
REAL_CSV = Path(r"C:\Users\qimen\Desktop\MILP scenario generation data\50 scenarios\csv\real_2024_2025.csv")
N_YEARS = 50
SYN = [f"n{i:02d}" for i in range(1, N_YEARS + 1)]
KNOB = [f"k{i:02d}" for i in range(1, N_YEARS + 1)]
SINGLE = [f"y{y}" for y in YEARS]
OWN = ["r2024", "r2025"]
H_YEAR = 8760
FIRST_WINDOW_DAY = -6  # window k starts 2022-12-26 + 12 k days; its centre 12 days are days 12k..12k+12 of 2023
SEASON_DAYS = 30


def knob_candidates(knobs, wind_h, time_h, start) -> dict:
    cum = np.concatenate([[0.0], np.cumsum(wind_h)])
    out = {}
    for k, seg in enumerate(range(0, H_YEAR, SEGMENT_H)):
        ctx = seg - SEGMENT_H // 2
        offset = 24 * (FIRST_WINDOW_DAY + 12 * k)
        a = np.array([start[y] + offset + 24 * j for y in YEARS for j in range(-SEASON_DAYS, SEASON_DAYS + 1)])
        a = a[(a >= 0) & (a + WINDOW_H <= len(wind_h))]
        a = a[(time_h[a + WINDOW_H - 1] - time_h[a] == WINDOW_H - 1) & (time_h[a] % 24 == 0)]
        mean = (cum[a + WINDOW_H] - cum[a]) / WINDOW_H
        out[seg] = (ctx, a[np.argsort(np.abs(mean - knobs[k]))[:TOP_MATCHES]])
    return out


def window_means(wind) -> np.ndarray:
    """24-day mean wind of the windows behind each 12-day segment (clipped to the year)."""

    return np.array([wind[max(0, s - SEGMENT_H // 2) : s - SEGMENT_H // 2 + WINDOW_H].mean() for s in range(0, H_YEAR, SEGMENT_H)])


def prepare(data_dir: str) -> None:
    d = Path(data_dir)
    real = pd.read_csv(REAL_CSV, parse_dates=["time_utc"])
    knobs = pd.read_csv(d / "knobs_per_window.csv", index_col=0)
    wind_h, k_h, time_h, start = load_history(real)
    clim = real.groupby([real["time_utc"].dt.dayofyear, real["time_utc"].dt.hour])["ghi_clearsky_wm2"].mean()
    arrays, err_n, err_k = {}, [], []
    for i in range(1, N_YEARS + 1):
        s = pd.read_csv(d / f"synthetic_year_{i:02d}.csv", parse_dates=["time_utc"])
        if len(s) != H_YEAR or s[["wind_ms", "ghi_wm2"]].isna().any().any() or (s[["wind_ms", "ghi_wm2"]] < 0).any().any():
            raise ValueError(f"synthetic_year_{i:02d}: unexpected length, gaps or negative values")
        cs = clim.reindex(pd.MultiIndex.from_arrays([s["time_utc"].dt.dayofyear, s["time_utc"].dt.hour])).to_numpy(float)
        kn = knobs.loc[f"synthetic_year_{i:02d}"].to_numpy(float)
        w_k, g_k = stitch(H_YEAR, knob_candidates(kn, wind_h, time_h, start), wind_h, k_h, np.nan_to_num(cs), seed=7000 + i)
        for name, w, g in ((f"n{i:02d}", s["wind_ms"].to_numpy(float), s["ghi_wm2"].to_numpy(float)), (f"k{i:02d}", w_k, g_k)):
            arrays[f"cfw_{name}"] = wind_power_cf(w)
            arrays[f"cfp_{name}"] = np.clip(PV_PR * g / 1000.0, 0.0, 1.0)
            arrays[f"wind_{name}"] = w.astype(np.float32)
        n_seg = len(window_means(s["wind_ms"].to_numpy(float)))
        err_n.append(window_means(s["wind_ms"].to_numpy(float)) - kn[:n_seg])
        err_k.append(window_means(w_k) - kn[:n_seg])
    for y in YEARS:
        h = power_year(y)
        arrays[f"cfw_y{y}"] = wind_power_cf(h["wind_ms"].to_numpy(float))
        arrays[f"cfp_y{y}"] = np.clip(PV_PR * h["ghi_wm2"].to_numpy(float) / 1000.0, 0.0, 1.0)
    cfw, cfp = load_series("real")
    n24 = int((real["time_utc"].dt.year == 2024).sum())
    arrays["cfw_r2024"], arrays["cfp_r2024"] = cfw[:n24], cfp[:n24]
    arrays["cfw_r2025"], arrays["cfp_r2025"] = cfw[n24:], cfp[n24:]
    np.savez_compressed(INPUTS, **arrays)
    en, ek = np.concatenate(err_n), np.concatenate(err_k)
    print("24-day window mean minus knob, m/s: synthetic bias %+.3f RMS %.3f | knob analogs bias %+.3f RMS %.3f" % (en.mean(), np.sqrt((en**2).mean()), ek.mean(), np.sqrt((ek**2).mean())))
    for label, names in (("synthetic", SYN), ("knob analogs", KNOB), ("single years", SINGLE), ("real years", OWN), ("real 2024-25", None)):
        if names is None:
            w, p = cfw.mean(), cfp.mean()
            print(f"{label:13s} wind CF {w:.4f}  PV CF {p:.4f}")
            continue
        w = np.array([arrays[f"cfw_{n}"].mean() for n in names])
        p = np.array([arrays[f"cfp_{n}"].mean() for n in names])
        print(f"{label:13s} wind CF {w.mean():.4f} (sd {w.std(ddof=1):.4f})  PV CF {p.mean():.4f} (sd {p.std(ddof=1):.4f})")
    ws = np.array([arrays[f"wind_{n}"].mean() for n in SYN])
    wk = np.array([arrays[f"wind_{n}"].mean() for n in KNOB])
    print("annual mean wind m/s: synthetic %.3f (sd %.3f), knob analogs %.3f (sd %.3f), paired corr %.2f" % (ws.mean(), ws.std(ddof=1), wk.mean(), wk.std(ddof=1), np.corrcoef(ws, wk)[0, 1]))


def load_synthetic(name: str):
    if name == "real":
        return load_series("real")
    with np.load(INPUTS) as z:
        return z[f"cfw_{name}"], z[f"cfp_{name}"]


def design_task(args) -> dict:
    name, kappa, max_wind, threads, limit = args
    case = OffgridCase(kappa=kappa, max_wind_MW=max_wind)
    des = design_lp(*load_synthetic(name), case, time_limit=limit, threads=threads)
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


def syn_path(tag: str) -> Path:
    return OUT / f"synthetic_design_{tag}.csv"


def syn_consensus_path(tag: str) -> Path:
    return OUT / f"synthetic_consensus_{tag}.csv"


def design(tag, kappa, max_wind, workers, threads, limit) -> None:
    path = syn_path(tag)
    done = set(pd.read_csv(path)["dataset"]) if path.exists() else set()
    order = [*OWN, *[x for pair in zip(SYN, KNOB) for x in pair], *SINGLE]
    names = [n for n in order if n not in done]
    print(f"{len(names)} designs to solve", flush=True)
    _pool(design_task, [(n, kappa, max_wind, threads, limit) for n in names], path, workers, lambda r: f"{r['dataset']} on real {r['e_TAC_usd_per_t']:.2f}")

    d = pd.read_csv(path).set_index("dataset")
    rng = np.random.default_rng(2026)
    keys = []
    for kind, names in (("syn", SYN), ("knob", KNOB)):
        for k in SUBSET_SIZES:
            for r in range(SUBSET_REPEATS):
                keys.append(({"kind": kind, "K": k, "rep": r}, mean_caps(d.loc[rng.choice(names, k, replace=False)])))
        keys.append(({"kind": kind, "K": N_YEARS, "rep": 0}, mean_caps(d.loc[names])))
    out = syn_consensus_path(tag)
    seen = set()
    if out.exists():
        old = pd.read_csv(out)
        seen = set(zip(old["kind"], old["K"], old["rep"]))
    tasks = [(key, caps, kappa, max_wind, threads, limit) for key, caps in keys if (key["kind"], key["K"], key["rep"]) not in seen]
    print(f"{len(tasks)} consensus tests to solve", flush=True)
    _pool(consensus_task, tasks, out, workers, lambda r: f"{r['kind']} K={r['K']} on real {r['TAC_usd_per_t']:.2f}")


def report(tag: str) -> None:
    ref = pd.read_csv(design_path(tag)).set_index("dataset")
    best = ref.loc["real", "e_TAC_usd_per_t"]
    d = pd.read_csv(syn_path(tag)).set_index("dataset")
    print("status:", d["d_status"].value_counts().to_dict(), d["e_status"].value_counts().to_dict(), f"n={len(d)}")
    pct = lambda s: 100.0 * (np.asarray(s, dtype=float) / best - 1.0)  # noqa: E731
    groups = {"synthetic years": SYN, "knob analogs": KNOB, "single 2005-23": SINGLE}
    print(f"\nReal-optimal cost {best:.2f} USD/t (two-year design); regret on real 2024-25 in %")
    for label, names in groups.items():
        print(f"  {label:16s}:", quant(pct(d.loc[names, "e_TAC_usd_per_t"])), f"(n={len(names)})")
    for n in OWN:
        print(f"  {n:16s}: {pct(d.loc[n, 'e_TAC_usd_per_t']):.2f}  (in-sample one-year basis)")
    rn, rk = pct(d.loc[SYN, "e_TAC_usd_per_t"]), pct(d.loc[KNOB, "e_TAC_usd_per_t"])
    diff = rn - rk
    boot = np.random.default_rng(0).choice(diff, (20000, len(diff))).mean(1)
    print("  paired synthetic - knob analog: mean %+.2f points, bootstrap 95%% CI %+.2f to %+.2f; Wilcoxon p = %.3f; synthetic better in %d/%d pairs"
          % (diff.mean(), *np.percentile(boot, [2.5, 97.5]), stats.wilcoxon(rn, rk).pvalue, (diff < 0).sum(), len(diff)))
    print("  synthetic vs single 2005-23 years: Mann-Whitney p = %.3f" % stats.mannwhitneyu(rn, pct(d.loc[SINGLE, "e_TAC_usd_per_t"])).pvalue)
    for label, names in groups.items():
        g = d.loc[names]
        print(f"  {label:16s}: shortfall > 1 t/y on real {(g.e_shortfall_t_y > 1).sum()}/{len(g)}, regret split fixed {g.e_fixed_usd_per_t.mean() - ref.loc['real', 'e_fixed_usd_per_t']:+.2f} / penalty {g.e_penalty_usd_per_t.mean() - ref.loc['real', 'e_penalty_usd_per_t']:+.2f} USD/t")

    print("\nDesigns: real two-year optimum | group mean (sd)")
    print("  %-14s %9s | %s" % ("", "real", " | ".join(f"{k:>20s}" for k in groups)))
    for c in ["C_wind_MW", "C_PV_MW", "C_PEM_MW", "C_BESS_MWh", "C_H2_kg", "TAC_usd_per_t"]:
        cells = " | ".join(f"{d.loc[n, f'd_{c}'].mean():10.2f} ({d.loc[n, f'd_{c}'].std(ddof=1):7.2f})" for n in groups.values())
        print(f"  {c:14s} {ref.loc['real', f'd_{c}']:9.2f} | {cells}")

    with np.load(INPUTS) as z:
        print("\nCapacity factors, group mean (sd)")
        for label, names in groups.items():
            w = np.array([z[f"cfw_{n}"].mean() for n in names])
            p = np.array([z[f"cfp_{n}"].mean() for n in names])
            print(f"  {label:16s} wind {w.mean():.4f} ({w.std(ddof=1):.4f})  PV {p.mean():.4f} ({p.std(ddof=1):.4f})")
    cfw, cfp = load_series("real")
    print(f"  {'real 2024-25':16s} wind {cfw.mean():.4f}           PV {cfp.mean():.4f}")

    c = pd.read_csv(syn_consensus_path(tag))
    print("\nConsensus regret on real, %: K | synthetic | knob analogs")
    print(f"  {1:2d} | {rn.mean():6.3f} | {rk.mean():6.3f}")
    for k in [*SUBSET_SIZES, N_YEARS]:
        a = pct(c.loc[(c.kind == "syn") & (c.K == k), "TAC_usd_per_t"]).mean()
        b = pct(c.loc[(c.kind == "knob") & (c.K == k), "TAC_usd_per_t"]).mean()
        print(f"  {k:2d} | {a:6.3f} | {b:6.3f}")


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

"""
Decision-based evaluation of generated wind and solar scenarios (off-grid PtM).

For each weather dataset X the plant is designed on X, the capacities are then
fixed, and the operation is re-optimized on the real series. The regret

    Regret_X = TAC(d_X; real) - TAC(d_real; real)

is the extra cost of having designed on X instead of on the real weather.

Datasets (2024-2025, hourly):
    real        observed wind speed (NASA POWER WS50M) and GHI
    gen         generated scenario under evaluation
    analog1..5  noise floor: real data from the other year, same season, chosen by
                the generator's conditioning (576-h mean wind) and stitched like it
                (centre 12 days, 6-h crossfade, clear-sky index x target clear-sky GHI)
    shuffle     real series with the days of each 576-h block permuted (no multi-day persistence)
    smooth6     real wind with a 6-h moving average (weaker ramps)
    ghi_bias    real GHI scaled by the generated/real mean ratio

Year-swap reference: a design from real 2024 tested on real 2025, and the reverse.

Usage:
    python evaluate_scenarios.py prepare --data <folder>
    python evaluate_scenarios.py run --workers 4 --threads 5
    python evaluate_scenarios.py verify --workers 3 --threads 6   (integer commitment)
    python evaluate_scenarios.py report
Use --tag, --kappa and --max-wind for variants.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from offgrid_design_gamspy import BARRIER, Capacities, OffgridCase, capacities_of, design_lp, solve_offgrid
from run_ptm_examples import RESULTS, wind_power_cf

OUT = RESULTS / "scenario_eval"
INPUTS = OUT / "inputs.npz"
WINDOW_H = 576
SEGMENT_H = 288  # generator stitching: one window every 12 days, centre 12 days kept
CROSSFADE_H = 6
N_ANALOG = 5
PV_PR = 0.86  # performance ratio on the horizontal plane (PVGIS default 14 % loss)
H_2024 = 8784
ANALOGS = [f"analog{i}" for i in range(1, N_ANALOG + 1)]
DATASETS = ["real", "gen", *ANALOGS, "shuffle", "smooth6", "ghi_bias"]


def clear_sky_index(ghi, cs):
    return np.clip(np.divide(ghi, cs, out=np.zeros_like(ghi), where=cs > 20.0), 0.0, 1.6)


def analog_series(wind, k, cs, seed: int):
    """Stitch real segments from the other year with the generator's conditioning."""

    rng = np.random.default_rng(seed)
    n = len(wind)
    cum = np.concatenate([[0.0], np.cumsum(wind)])
    mean_w = lambda a: (cum[a + WINDOW_H] - cum[a]) / WINDOW_H  # noqa: E731
    w_sum, k_sum, weight = np.zeros(n), np.zeros(n), np.zeros(n)
    half = CROSSFADE_H // 2
    for seg in range(0, n, SEGMENT_H):
        ctx = int(np.clip(seg + SEGMENT_H // 2 - WINDOW_H // 2, 0, n - WINDOW_H))
        shift = H_2024 if ctx + WINDOW_H // 2 < H_2024 else -H_2024
        cands = [ctx + shift + 24 * j for j in range(-30, 31)]
        cands = [a for a in cands if 0 <= a <= n - WINDOW_H]
        cands.sort(key=lambda a: abs(mean_w(a) - mean_w(ctx)))
        a = cands[int(rng.integers(0, min(5, len(cands))))]
        idx = np.arange(max(0, seg - half), min(n, seg + SEGMENT_H + half))
        ramp = np.minimum.reduce(
            [np.ones(len(idx)), (idx - (seg - half) + 0.5) / CROSSFADE_H, (seg + SEGMENT_H + half - idx - 0.5) / CROSSFADE_H]
        )
        src = np.clip(a + idx - ctx, 0, n - 1)
        w_sum[idx] += ramp * wind[src]
        k_sum[idx] += ramp * k[src]
        weight[idx] += ramp
    return w_sum / weight, (k_sum / weight) * cs


def shuffle_days(wind, k, seed: int):
    rng = np.random.default_rng(seed)
    w2, k2 = wind.copy(), k.copy()
    for s in range(0, len(wind) - WINDOW_H + 1, WINDOW_H):
        perm = rng.permutation(WINDOW_H // 24)
        pos = np.concatenate([np.arange(24 * p, 24 * p + 24) for p in perm]) + s
        w2[s : s + WINDOW_H], k2[s : s + WINDOW_H] = wind[pos], k[pos]
    return w2, k2


def prepare(data_dir: str) -> None:
    d = Path(data_dir)
    real = pd.read_csv(d / "real_2024_2025.csv", parse_dates=["time_utc"])
    gen = pd.read_csv(d / "generated_E5_2024_2025.csv", parse_dates=["time_utc"])
    if not (real["time_utc"] == gen["time_utc"]).all():
        raise ValueError("Real and generated time stamps differ.")
    w, g, cs = (real[c].to_numpy(float) for c in ("wind_ms", "ghi_wm2", "ghi_clearsky_wm2"))
    k = clear_sky_index(g, cs)
    sets = {"real": (w, g), "gen": (gen["wind_ms"].to_numpy(float), gen["ghi_wm2"].to_numpy(float))}
    for i in range(1, N_ANALOG + 1):
        sets[f"analog{i}"] = analog_series(w, k, cs, seed=100 + i)
    ws, ks = shuffle_days(w, k, seed=7)
    sets["shuffle"] = (ws, ks * cs)
    sets["smooth6"] = (pd.Series(w).rolling(6, center=True, min_periods=1).mean().to_numpy(), g)
    sets["ghi_bias"] = (w, g * sets["gen"][1].mean() / g.mean())

    arrays = {}
    for name, (wind, ghi) in sets.items():
        arrays[f"cfw_{name}"] = wind_power_cf(wind)
        arrays[f"cfp_{name}"] = np.clip(PV_PR * ghi / 1000.0, 0.0, 1.0)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(INPUTS, **arrays)
    print(f"{'dataset':10s} {'wind CF':>8s} {'PV CF':>7s}")
    for name in sets:
        print(f"{name:10s} {arrays['cfw_' + name].mean():8.4f} {arrays['cfp_' + name].mean():7.4f}")


def span_slice(span: str) -> slice:
    return {"2y": slice(None), "2024": slice(0, H_2024), "2025": slice(H_2024, None)}[span]


def tasks_for(kappas) -> list:
    tasks = [(kappa, d, "2y", "2y") for kappa in kappas for d in DATASETS]
    for kappa in kappas:
        tasks += [(kappa, "real", "2024", "2025"), (kappa, "real", "2025", "2024")]
        tasks += [(kappa, "real", "2024", "2024"), (kappa, "real", "2025", "2025")]
    return tasks


def solve_task(args) -> dict:
    (kappa, dataset, span, test), threads, limit, max_wind = args
    with np.load(INPUTS) as z:
        w, p = z[f"cfw_{dataset}"][span_slice(span)], z[f"cfp_{dataset}"][span_slice(span)]
        w_real, p_real = z["cfw_real"][span_slice(test)], z["cfp_real"][span_slice(test)]
    case = OffgridCase(kappa=kappa, max_wind_MW=max_wind)
    des = design_lp(w, p, case, time_limit=limit, threads=threads)
    ev = solve_offgrid(
        w_real, p_real, case, fixed=capacities_of(des), integer=False,
        time_limit=limit, threads=threads, solver_options=BARRIER,
    )
    row = {"kappa": kappa, "dataset": dataset, "design_span": span, "test_span": test}
    row.update({f"d_{k}": v for k, v in des.items()})
    row.update({f"e_{k}": v for k, v in ev.items() if k not in ("C_wind_MW", "C_PV_MW", "N_PEM", "C_BESS_MWh", "C_H2_kg", "C_PEM_MW")})
    return row


def result_path(tag: str) -> Path:
    return OUT / f"offgrid_design{'_' + tag if tag else ''}.csv"


def verify_task(args) -> dict:
    """Test a stored design on the real series with integer module commitment."""

    row, threads, limit, gap, max_wind = args
    with np.load(INPUTS) as z:
        w, p = z["cfw_real"], z["cfp_real"]
    caps = Capacities(row["d_C_wind_MW"], row["d_C_PV_MW"], row["d_N_PEM"], row["d_C_BESS_MWh"], row["d_C_H2_kg"])
    case = OffgridCase(kappa=row["kappa"], max_wind_MW=max_wind)
    ev = solve_offgrid(w, p, case, fixed=caps, integer=True, gap=gap, time_limit=limit, threads=threads)
    return {"kappa": row["kappa"], "dataset": row["dataset"], "TAC_relaxed": row["e_TAC_usd_per_t"],
            "TAC_integer": ev["TAC_usd_per_t"], "status": ev["status"], "solve_s": ev["solve_s"],
            "PEM_starts_y": ev["PEM_starts_y"], "unserved_MWh_y": ev["unserved_MWh_y"], "shortfall_t_y": ev["shortfall_t_y"]}


def verify(tag: str, workers: int, threads: int, limit: float, gap: float, max_wind: float) -> None:
    df = pd.read_csv(result_path(tag), dtype={"design_span": str, "test_span": str})
    two = df[(df["design_span"] == "2y") & df["dataset"].isin(["real", "gen", *ANALOGS])]
    path = OUT / f"offgrid_integer{'_' + tag if tag else ''}.csv"
    with ProcessPoolExecutor(workers) as ex:
        futures = [ex.submit(verify_task, (r.to_dict(), threads, limit, gap, max_wind)) for _, r in two.iterrows()]
        for fut in as_completed(futures):
            row = fut.result()
            pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)
            print(row, flush=True)


def run(kappas, workers: int, threads: int, limit: float, tag: str, max_wind: float) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = result_path(tag)
    done = set()
    if path.exists():
        old = pd.read_csv(path)
        done = set(zip(old["kappa"], old["dataset"], old["design_span"].astype(str), old["test_span"].astype(str)))
    todo = [t for t in tasks_for(kappas) if t not in done]
    print(f"{len(todo)} design + test pairs to solve, {len(done)} done", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        futures = {ex.submit(solve_task, (t, threads, limit, max_wind)): t for t in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)
            print(f"  {i}/{len(todo)} {futures[fut]} TAC on real {row['e_TAC_usd_per_t']:.2f} USD/t  {time.time() - t0:.0f} s", flush=True)


def report(tag: str) -> None:
    df = pd.read_csv(result_path(tag), dtype={"design_span": str, "test_span": str})
    pd.set_option("display.width", 240, "display.max_columns", 40, "display.float_format", lambda v: f"{v:,.2f}")
    print("status design:", df["d_status"].value_counts().to_dict(), "| test:", df["e_status"].value_counts().to_dict())
    print("solve time [s]: design median %.0f max %.0f; test median %.0f max %.0f" % (
        df["d_solve_s"].median(), df["d_solve_s"].max(), df["e_solve_s"].median(), df["e_solve_s"].max()))
    rows = []
    for kappa, g in df.groupby("kappa"):
        two = g[(g["design_span"] == "2y") & (g["test_span"] == "2y")].set_index("dataset")
        best = two.loc["real", "e_TAC_usd_per_t"]
        for d, r in two.iterrows():
            rows.append(dict(
                kappa=kappa, dataset=d, TAC_on_own=r["d_TAC_usd_per_t"], TAC_on_real=r["e_TAC_usd_per_t"],
                regret=r["e_TAC_usd_per_t"] - best, regret_pct=100 * (r["e_TAC_usd_per_t"] / best - 1),
                wind_MW=r["d_C_wind_MW"], pv_MW=r["d_C_PV_MW"], pem_MW=r["d_C_PEM_MW"], bess_MWh=r["d_C_BESS_MWh"],
                h2_t=r["d_C_H2_kg"] / 1000.0, curtail_on_real=r["e_curtail_share"],
                unserved_MWh_y=r["e_unserved_MWh_y"], shortfall_t_y=r["e_shortfall_t_y"],
            ))
        for test, train in (("2025", "2024"), ("2024", "2025")):
            ref = g[(g["design_span"] == test) & (g["test_span"] == test)]["e_TAC_usd_per_t"].iloc[0]
            r = g[(g["design_span"] == train) & (g["test_span"] == test)].iloc[0]
            rows.append(dict(
                kappa=kappa, dataset=f"real{train}->real{test}", TAC_on_own=r["d_TAC_usd_per_t"], TAC_on_real=r["e_TAC_usd_per_t"],
                regret=r["e_TAC_usd_per_t"] - ref, regret_pct=100 * (r["e_TAC_usd_per_t"] / ref - 1),
                wind_MW=r["d_C_wind_MW"], pv_MW=r["d_C_PV_MW"], pem_MW=r["d_C_PEM_MW"], bess_MWh=r["d_C_BESS_MWh"],
                h2_t=r["d_C_H2_kg"] / 1000.0, curtail_on_real=r["e_curtail_share"],
                unserved_MWh_y=r["e_unserved_MWh_y"], shortfall_t_y=r["e_shortfall_t_y"],
            ))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / f"report_offgrid{'_' + tag if tag else ''}.csv", index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["prepare", "run", "verify", "report"])
    ap.add_argument("--data", help="folder with real_2024_2025.csv and generated_E5_2024_2025.csv")
    ap.add_argument("--kappa", type=float, nargs="+", default=[1.25])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=5)
    ap.add_argument("--time-limit", type=float, default=3600.0)
    ap.add_argument("--tag", default="", help="suffix of the result files")
    ap.add_argument("--max-wind", type=float, default=500.0, help="upper bound on wind capacity [MW]")
    ap.add_argument("--gap", type=float, default=5e-4, help="relative MIP gap in verify")
    args = ap.parse_args()
    if args.command == "prepare":
        prepare(args.data)
    elif args.command == "run":
        run(args.kappa, args.workers, args.threads, args.time_limit, args.tag, args.max_wind)
    elif args.command == "verify":
        verify(args.tag, args.workers, args.threads, args.time_limit, args.gap, args.max_wind)
    else:
        report(args.tag)

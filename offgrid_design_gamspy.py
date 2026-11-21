"""
GAMSPy design and operation MILP for an off-grid wind + PV power-to-methanol plant.

Capacities (wind, PV, PEM modules, battery, H2 tank) and the hourly schedule are
optimized together under perfect foresight, minimizing total annual cost. With
`fixed` capacities the same model gives the optimal operation of a given design,
which is how a design is tested on another weather series.

Modelling choices:
    - No grid. Unserved electricity is a slack with a high penalty, so that any
      design remains solvable on any weather series and its adequacy can be measured.
    - The electrolyzer is built from identical modules. Installed, running and
      standby module counts are integers, which keeps the model linear when the
      capacity is a decision variable (no capacity x binary products).
    - Methanol synthesis stays on between its minimum load and nameplate. Nameplate
      is kappa x requirement and is fixed here, so synthesis and distillation CAPEX
      are a constant left out of the objective.
    - Storage levels, commitment and synthesis load are cyclic over the horizon.

Technical parameters come from OperatingParams in ptm_scheduling_gamspy.py; cost
data are those of MasCOR (solvers/GLOBAL_solver.py).

Units: h, MW, MWh, kg/h, kg, USD. Objective in USD/year.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Optional

import gamspy as gp
import numpy as np

from ptm_scheduling_gamspy import OperatingParams, pem_curve_segments


@dataclass
class OffgridCase:
    D_MeOH_kgph: float = 1000.0  # average methanol requirement
    kappa: float = 1.25  # synthesis nameplate / requirement
    M_PEM_MW: float = 2.5  # electrolyzer module size
    c_unserved: float = 2000.0  # USD/MWh unserved electricity. ASSUMPTION
    max_wind_MW: float = 500.0
    max_pv_MW: float = 500.0
    max_PEM_modules: int = 80


@dataclass
class Costs:
    wind: float = 1250e3  # USD/MW
    pv: float = 740e3  # USD/MW
    pem: float = 600e3  # USD/MW
    h2: float = 751.7  # USD/kg
    bess: float = 236.5e3  # USD/MWh
    om_wind: float = 25e3  # USD/MW/year
    om_pv: float = 12.6e3  # USD/MW/year
    rate: float = 0.08
    years: int = 25

    @property
    def crf(self) -> float:
        return self.rate * (1 + self.rate) ** self.years / ((1 + self.rate) ** self.years - 1)


@dataclass
class Capacities:
    C_wind_MW: float
    C_PV_MW: float
    N_PEM: float
    C_BESS_MWh: float
    C_H2_kg: float


def annual_fixed_cost(c: Capacities, case: OffgridCase, cost: Costs) -> float:
    capex = (
        cost.wind * c.C_wind_MW
        + cost.pv * c.C_PV_MW
        + cost.pem * case.M_PEM_MW * c.N_PEM
        + cost.h2 * c.C_H2_kg
        + cost.bess * c.C_BESS_MWh
    )
    return cost.crf * capex + cost.om_wind * c.C_wind_MW + cost.om_pv * c.C_PV_MW


def solve_offgrid(
    cf_wind,
    cf_pv,
    case: Optional[OffgridCase] = None,
    par: Optional[OperatingParams] = None,
    cost: Optional[Costs] = None,
    fixed: Optional[Capacities] = None,
    fix_N: Optional[float] = None,
    integer: bool = True,
    gap: float = 1e-3,
    time_limit: float = 3600.0,
    threads: int = 4,
    solver_options: Optional[dict] = None,
) -> dict:
    """Build and solve the model; return capacities, annual costs and operating results."""

    case = case or OffgridCase()
    par = par or OperatingParams()
    cost = cost or Costs()
    cf_wind, cf_pv = np.asarray(cf_wind, dtype=float), np.asarray(cf_pv, dtype=float)
    n_t = len(cf_wind)
    dt = par.dt_h
    annual = 8760.0 / (n_t * dt)
    m_nom = case.kappa * case.D_MeOH_kgph
    M = case.M_PEM_MW
    slopes, intercepts = pem_curve_segments(par.pem_curve)

    m = gp.Container()
    steps = [f"t{i:05d}" for i in range(1, n_t + 1)]
    t = gp.Set(m, "t", records=steps)
    k = gp.Set(m, "k", records=[f"k{i}" for i in range(len(slopes))])
    cfw = gp.Parameter(m, "cfw", domain=t, records=cf_wind)
    cfp = gp.Parameter(m, "cfp", domain=t, records=cf_pv)
    slope = gp.Parameter(m, "slope", domain=k, records=slopes)
    icpt = gp.Parameter(m, "icpt", domain=k, records=intercepts)
    prev = t.lag(1, "circular")

    # Design variables
    C_W = gp.Variable(m, "C_W", type="Positive")
    C_P = gp.Variable(m, "C_P", type="Positive")
    C_B = gp.Variable(m, "C_B", type="Positive")
    C_H = gp.Variable(m, "C_H", type="Positive")
    int_type = "Integer" if integer else "Positive"
    N = gp.Variable(m, "N", type=int_type)
    C_W.up[...] = case.max_wind_MW
    C_P.up[...] = case.max_pv_MW
    N.up[...] = case.max_PEM_modules
    N.lo[...] = 1
    if fixed is not None:
        C_W.fx[...] = fixed.C_wind_MW
        C_P.fx[...] = fixed.C_PV_MW
        C_B.fx[...] = fixed.C_BESS_MWh
        C_H.fx[...] = fixed.C_H2_kg
        N.fx[...] = fixed.N_PEM
    elif fix_N is not None:
        N.fx[...] = fix_N

    # Operating variables
    P_curt = gp.Variable(m, "P_curt", domain=t, type="Positive")
    P_uns = gp.Variable(m, "P_uns", domain=t, type="Positive")
    P_ch = gp.Variable(m, "P_ch", domain=t, type="Positive")
    P_dis = gp.Variable(m, "P_dis", domain=t, type="Positive")
    E = gp.Variable(m, "E", domain=t, type="Positive")
    P_PEM = gp.Variable(m, "P_PEM", domain=t, type="Positive")
    P_comp = gp.Variable(m, "P_comp", domain=t, type="Positive")
    m_H2 = gp.Variable(m, "m_H2", domain=t, type="Positive")
    h_in = gp.Variable(m, "h_in", domain=t, type="Positive")
    h_out = gp.Variable(m, "h_out", domain=t, type="Positive")
    H = gp.Variable(m, "H", domain=t, type="Positive")
    m_M = gp.Variable(m, "m_M", domain=t, type="Positive")
    n_on = gp.Variable(m, "n_on", domain=t, type=int_type)
    n_sb = gp.Variable(m, "n_sb", domain=t, type=int_type)
    n_up = gp.Variable(m, "n_up", domain=t, type="Positive")
    short = gp.Variable(m, "short", type="Positive")
    n_on.up[t] = case.max_PEM_modules
    n_sb.up[t] = case.max_PEM_modules

    # Electricity
    gen = cfw[t] * C_W + cfp[t] * C_P
    eq_power = gp.Equation(m, "eq_power", domain=t)
    eq_power[t] = gen - P_curt[t] + P_dis[t] + P_uns[t] == (
        P_ch[t] + P_PEM[t] + par.f_sb_PEM * M * n_sb[t] + P_comp[t] + par.SP_MeOH / 1000.0 * m_M[t]
    )
    eq_curt = gp.Equation(m, "eq_curt", domain=t)
    eq_curt[t] = P_curt[t] <= gen

    # Battery
    eq_soc = gp.Equation(m, "eq_soc", domain=t)
    eq_soc[t] = E[t] == (1.0 - par.loss_BESS) * E[prev] + (par.eta_ch * P_ch[t] - P_dis[t] / par.eta_dis) * dt
    eq_soc_lo = gp.Equation(m, "eq_soc_lo", domain=t)
    eq_soc_lo[t] = E[t] >= par.soc_min * C_B
    eq_soc_up = gp.Equation(m, "eq_soc_up", domain=t)
    eq_soc_up[t] = E[t] <= par.soc_max * C_B
    eq_ch = gp.Equation(m, "eq_ch", domain=t)
    eq_ch[t] = P_ch[t] <= par.c_rate * C_B
    eq_dis = gp.Equation(m, "eq_dis", domain=t)
    eq_dis[t] = P_dis[t] <= par.c_rate * C_B

    # Electrolyzer modules
    eq_mod = gp.Equation(m, "eq_mod", domain=t)
    eq_mod[t] = n_on[t] + n_sb[t] <= N
    eq_pem_lo = gp.Equation(m, "eq_pem_lo", domain=t)
    eq_pem_lo[t] = P_PEM[t] >= par.f_min_PEM * M * n_on[t]
    eq_pem_up = gp.Equation(m, "eq_pem_up", domain=t)
    eq_pem_up[t] = P_PEM[t] <= M * n_on[t]
    eq_start = gp.Equation(m, "eq_start", domain=t)
    eq_start[t] = n_up[t] >= n_on[t] - n_on[prev] - n_sb[prev]
    eq_curve = gp.Equation(m, "eq_curve", domain=[k, t])
    eq_curve[k, t] = m_H2[t] <= slope[k] * P_PEM[t] + icpt[k] * M * n_on[t]

    # Hydrogen
    eq_h2 = gp.Equation(m, "eq_h2", domain=t)
    eq_h2[t] = m_H2[t] + h_out[t] == par.gamma_H2 * m_M[t] + h_in[t]
    eq_tank = gp.Equation(m, "eq_tank", domain=t)
    eq_tank[t] = H[t] == H[prev] + (h_in[t] - h_out[t]) * dt
    eq_tank_up = gp.Equation(m, "eq_tank_up", domain=t)
    eq_tank_up[t] = H[t] <= C_H
    eq_comp = gp.Equation(m, "eq_comp", domain=t)
    eq_comp[t] = P_comp[t] == par.SP_comp / 1000.0 * h_in[t]

    # Methanol synthesis
    m_M.lo[t] = par.f_min_MeOH * m_nom
    m_M.up[t] = m_nom
    eq_ramp_up = gp.Equation(m, "eq_ramp_up", domain=t)
    eq_ramp_up[t] = m_M[t] - m_M[prev] <= par.ramp_MeOH * m_nom * dt
    eq_ramp_dn = gp.Equation(m, "eq_ramp_dn", domain=t)
    eq_ramp_dn[t] = m_M[prev] - m_M[t] <= par.ramp_MeOH * m_nom * dt
    eq_target = gp.Equation(m, "eq_target")
    eq_target[...] = gp.Sum(t, m_M[t] * dt) / 1000.0 + short >= case.D_MeOH_kgph * n_t * dt / 1000.0

    # Objective, USD/year
    capex = cost.wind * C_W + cost.pv * C_P + cost.pem * M * N + cost.h2 * C_H + cost.bess * C_B
    fixed_cost = cost.crf * capex + cost.om_wind * C_W + cost.om_pv * C_P
    variable = gp.Sum(
        t,
        (
            par.c_water_H2 * m_H2[t]
            + par.c_CO2_feed * par.gamma_CO2 / 1000.0 * m_M[t]
            + par.c_tiebreak_PEM * P_PEM[t]
            + case.c_unserved * P_uns[t]
        )
        * dt,
    ) + par.c_cold_start_PEM * M * gp.Sum(t, n_up[t]) + par.c_short_MeOH * short
    total = fixed_cost + annual * variable

    model = gp.Model(
        m,
        name="ptm_offgrid",
        equations=m.getEquations(),
        problem="MIP" if integer else "LP",
        sense=gp.Sense.MIN,
        objective=total,
    )
    t0 = time.perf_counter()
    summary = model.solve(
        solver="cplex",
        options=gp.Options(relative_optimality_gap=gap, time_limit=time_limit, threads=threads),
        solver_options=solver_options,
    )
    wall = time.perf_counter() - t0

    caps = Capacities(C_W.toValue(), C_P.toValue(), N.toValue(), C_B.toValue(), C_H.toValue())
    caps = Capacities(*(float(x or 0.0) for x in asdict(caps).values()))

    def series(v):
        d = v.toDense()
        return np.zeros(n_t) if d is None else np.asarray(d, dtype=float)

    uns, curt, dis, out, pem, mh2, mm, on, sb, up = map(
        series, (P_uns, P_curt, P_dis, h_out, P_PEM, m_H2, m_M, n_on, n_sb, n_up)
    )
    short_t = float(short.toValue() or 0.0)
    g = cf_wind * caps.C_wind_MW + cf_pv * caps.C_PV_MW
    t_year = case.D_MeOH_kgph * 8760.0 / 1000.0
    fixed_usd = annual_fixed_cost(caps, case, cost)
    return {
        "TAC_usd": model.objective_value,
        "TAC_usd_per_t": model.objective_value / t_year,
        "fixed_usd_per_t": fixed_usd / t_year,
        "penalty_usd_per_t": annual * (case.c_unserved * uns.sum() * dt + par.c_short_MeOH * short_t) / t_year,
        **asdict(caps),
        "C_PEM_MW": caps.N_PEM * M,
        "gen_GWh_y": annual * g.sum() * dt / 1000.0,
        "wind_share": float((cf_wind * caps.C_wind_MW).sum() / max(g.sum(), 1e-9)),
        "curtail_share": float(curt.sum() / max(g.sum(), 1e-9)),
        "unserved_MWh_y": annual * uns.sum() * dt,
        "unserved_hours_y": annual * float((uns > 1e-3).sum()) * dt,
        "shortfall_t_y": annual * short_t,
        "BESS_MWh_y": annual * dis.sum() * dt,
        "H2_tank_t_y": annual * out.sum() * dt / 1000.0,
        "PEM_FLH": annual * pem.sum() * dt / max(caps.N_PEM * M, 1e-9),
        "PEM_starts_y": annual * up.sum(),
        "PEM_standby_module_h_y": annual * sb.sum() * dt,
        "SEC_kWh_kg": 1000.0 * pem.sum() / max(mh2.sum(), 1e-9),
        "MeOH_load_mean": float(mm.mean() / m_nom),
        "MeOH_load_std": float((mm / m_nom).std()),
        "status": str(summary.iloc[0]["Model Status"]),
        "solve_s": wall,
    }


BARRIER = {"lpmethod": 4, "barcrossalg": -1}  # interior point without crossover


def capacities_of(res: dict) -> Capacities:
    return Capacities(res["C_wind_MW"], res["C_PV_MW"], res["N_PEM"], res["C_BESS_MWh"], res["C_H2_kg"])


def design_and_operate(
    cf_wind,
    cf_pv,
    case: Optional[OffgridCase] = None,
    par: Optional[OperatingParams] = None,
    cost: Optional[Costs] = None,
    gap: float = 5e-4,
    time_limit: float = 1800.0,
    threads: int = 4,
) -> dict:
    """Solve the joint design and scheduling MILP by LP-based decomposition.

    The monolithic MILP does not converge on multi-year horizons, so:
      1. the LP relaxation gives a lower bound and the module count N_LP;
      2. capacities and schedule are optimized jointly for floor(N_LP) and ceil(N_LP);
      3. with those capacities fixed, module commitment is solved with integers.
    The best integer solution is returned with its gap to the LP lower bound, which
    bounds its distance from the true MILP optimum.
    """

    case = case or OffgridCase()
    lp_opts = dict(integer=False, time_limit=time_limit, threads=threads, solver_options=BARRIER)
    t0 = time.perf_counter()
    root = solve_offgrid(cf_wind, cf_pv, case, par, cost, **lp_opts)
    n_lp = root["N_PEM"]
    candidates = sorted({max(1.0, np.floor(n_lp + 1e-6)), max(1.0, np.ceil(n_lp - 1e-6))})
    best = None
    for n in candidates:
        lp = solve_offgrid(cf_wind, cf_pv, case, par, cost, fix_N=n, **lp_opts)
        mip = solve_offgrid(
            cf_wind, cf_pv, case, par, cost, fixed=capacities_of(lp), integer=True,
            gap=gap, time_limit=time_limit, threads=threads,
        )
        if best is None or mip["TAC_usd"] < best["TAC_usd"]:
            best = mip
    best["LB_usd_per_t"] = root["TAC_usd_per_t"]
    best["gap_to_LB_pct"] = 100.0 * (best["TAC_usd_per_t"] / root["TAC_usd_per_t"] - 1.0)
    best["N_LP"] = n_lp
    best["total_s"] = time.perf_counter() - t0
    return best


def design_lp(
    cf_wind,
    cf_pv,
    case: Optional[OffgridCase] = None,
    par: Optional[OperatingParams] = None,
    cost: Optional[Costs] = None,
    time_limit: float = 3600.0,
    threads: int = 4,
) -> dict:
    """Joint design and scheduling with integer module count and relaxed hourly commitment."""

    case = case or OffgridCase()
    opts = dict(integer=False, time_limit=time_limit, threads=threads, solver_options=BARRIER)
    root = solve_offgrid(cf_wind, cf_pv, case, par, cost, **opts)
    n_lp = root["N_PEM"]
    candidates = sorted({max(1.0, np.floor(n_lp + 1e-6)), max(1.0, np.ceil(n_lp - 1e-6))})
    runs = [solve_offgrid(cf_wind, cf_pv, case, par, cost, fix_N=n, **opts) for n in candidates]
    best = min(runs, key=lambda r: r["TAC_usd"])
    best["LB_usd_per_t"] = root["TAC_usd_per_t"]
    best["N_LP"] = n_lp
    return best

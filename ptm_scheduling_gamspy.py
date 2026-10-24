"""
GAMSPy MILP for hourly scheduling of a wind-powered power-to-methanol (PtM)
plant with bidirectional grid exchange, a battery (BESS) and a compressed H2
tank.

Process boundary:
    wind farm + grid -> PEM electrolyzer -> (H2 tank) -> methanol synthesis
    BESS on the electricity bus; optional H2 sales to an external market

Changes relative to the MasCOR operational model (Kim et al., arXiv:2603.03484,
mascor/solvers/GLOBAL_solver.py):
    1. PEM electrolyzer has on / warm-standby / off states, a minimum load,
       standby power and a cold-start cost. The part-load H2 curve is concave
       and is represented by its piecewise-linear upper envelope, so it needs
       no extra binaries.
    2. Methanol synthesis is flexible: on/off, minimum load, ramp limits,
       minimum up/down times and a start-up cost. MasCOR runs it at a constant
       nameplate rate.
    3. Wind curtailment is a decision variable and the import price can carry a
       grid fee, so the binary grid/BESS exclusivity products of MasCOR are not
       needed (an optional BESS charge/discharge binary is kept).
    4. Hourly grid emission factor and a carbon-intensity cap on methanol.
       CI_max = 1.435802 tCO2/tMeOH reproduces MasCOR's "net-negative" rule.
    5. A methanol production target (soft, with shortfall penalty) replaces
       unlimited H2 sales as the reason to run the plant.
    6. Terminal storage conditions remove end-of-horizon depletion.

Units used in this model:
    time: h
    power: MW, electrical energy: MWh
    H2 and methanol flows: kg/h, H2 inventory: kg
    electricity price: USD/MWh, grid emission factor: tCO2/MWh
    methanol, CO2 and emissions in cost/balance totals: t
"""

from __future__ import annotations

import functools
import operator
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

try:
    import gamspy as gp
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "This script requires GAMSPy. Install it with `pip install gamspy` "
        "and ensure that a GAMS license/runtime is available."
    ) from exc


@dataclass
class PlantDesign:
    """Installed capacities. Fixed here; first-stage decisions in design studies."""

    C_wind_MW: float = 50.0
    C_PEM_MW: float = 20.0
    C_BESS_MWh: float = 20.0
    C_H2_kg: float = 1000.0
    m_MeOH_nom_kgph: float = 1000.0
    G_max_MW: float = 50.0


@dataclass
class OperatingParams:
    """Technical and economic parameters.

    Values marked ASSUMPTION are placeholders that still need plant- or
    literature-specific calibration.
    """

    dt_h: float = 1.0

    # PEM electrolyzer
    f_min_PEM: float = 0.10  # minimum load fraction; PEM operates at 10-100 %
    f_sb_PEM: float = 0.02  # warm-standby power fraction (~2 % of rated)
    # (load fraction, system specific energy consumption kWh/kg H2).
    # Rated point 55.7 kWh/kg from MasCOR; shape follows the reported system
    # efficiency peak near 20 % load. ASSUMPTION for the intermediate points.
    pem_curve: tuple = ((0.10, 52.5), (0.20, 50.0), (0.50, 52.0), (1.00, 55.7))
    c_cold_start_PEM: float = 50.0  # USD per MW installed per cold start. ASSUMPTION
    # Tie-breaker: at zero price, running the stack below its H2 curve (i.e.
    # wasting power) is otherwise as cheap as exporting or curtailing.
    c_tiebreak_PEM: float = 0.01  # USD/MWh

    # Methanol synthesis
    f_min_MeOH: float = 0.20  # minimum load fraction (adiabatic / water-cooled reactors)
    ramp_MeOH: float = 0.50  # fraction of nameplate per h (50 -> 100 % in 1 h)
    UT_MeOH: int = 8  # minimum up time [h]. ASSUMPTION
    DT_MeOH: int = 4  # minimum down time [h]. ASSUMPTION
    c_start_MeOH: float = 5000.0  # USD per start. ASSUMPTION
    SP_MeOH: float = 0.65702  # kWh/kg MeOH (MasCOR)
    gamma_H2: float = 0.19576  # kg H2/kg MeOH (MasCOR)
    gamma_CO2: float = 1.435802  # kg CO2/kg MeOH (MasCOR)

    # H2 compression and storage
    SP_comp: float = 3.03  # kWh/kg H2 compressed into the tank (MasCOR)
    f_min_H2: float = 0.0  # minimum tank fraction (cushion gas)

    # BESS (MasCOR values)
    eta_ch: float = 0.95
    eta_dis: float = 0.95
    loss_BESS: float = 0.05 / 720.0  # self-discharge per h
    soc_min: float = 0.10
    soc_max: float = 0.90
    c_rate: float = 0.30  # power / energy [1/h]
    bess_exclusive: bool = False  # add charge/discharge binary

    # Economics
    grid_fee: float = 0.0  # USD/MWh added to the import price
    tax_CO2: float = 28.10  # USD/tCO2 on import emissions (Denmark, MasCOR Table S2)
    c_CO2_feed: float = 50.0  # USD/t captured CO2 feed (MasCOR)
    c_water_H2: float = 0.127  # USD/kg H2 water/consumables (MasCOR)
    p_H2_sale: float = 0.0  # USD/kg; MasCOR uses 5.0
    H2_sale_max_kgph: float = 0.0  # 0 disables H2 sales
    p_MeOH: float = 0.0  # USD/t methanol revenue
    target_frac_MeOH: float = 0.90  # required production / (nameplate * horizon)
    c_short_MeOH: float = 5000.0  # USD/t production shortfall
    CI_max: Optional[float] = 1.435802  # tCO2/tMeOH on grid imports; None disables

    # Initial and terminal conditions
    soc0: float = 0.50  # BESS SOC at t=0 [-]
    h0: float = 0.50  # H2 tank level at t=0 [-]
    u0_PEM: int = 1
    s0_PEM: int = 0
    u0_MeOH: int = 1
    m0_MeOH_frac: float = 1.0
    terminal_storage: bool = True  # end SOC and H2 level >= initial


@dataclass
class SchedulingData:
    """Hourly exogenous inputs."""

    time_steps: list[str]
    cf_wind: np.ndarray  # wind capacity factor [-]
    price: np.ndarray  # electricity price [USD/MWh]
    eps_grid: np.ndarray  # grid emission factor [tCO2/MWh]

    @classmethod
    def from_arrays(cls, cf_wind, price, eps_grid=0.5) -> "SchedulingData":
        cf_wind = np.asarray(cf_wind, dtype=float)
        n = len(cf_wind)
        eps = np.broadcast_to(np.asarray(eps_grid, dtype=float), (n,)).copy()
        return cls([f"t{i:04d}" for i in range(1, n + 1)], cf_wind, np.asarray(price, dtype=float), eps)


def pem_curve_segments(curve) -> tuple[np.ndarray, np.ndarray]:
    """Secant lines of the H2 production curve g(L) = 1000 L / SEC(L).

    g is in kg/h per MW installed. For a concave g, the minimum over the
    secant lines equals the piecewise-linear interpolant, so
        m_H2 <= slope_k * P_PEM + icpt_k * C_PEM * u_PEM   for all k
    is exact without SOS2 binaries.
    """

    L = np.array([p[0] for p in curve], dtype=float)
    sec = np.array([p[1] for p in curve], dtype=float)
    g = 1000.0 * L / sec
    slopes = np.diff(g) / np.diff(L)
    if np.any(np.diff(slopes) > 1e-12):
        raise ValueError("PEM H2 production curve is not concave; use SOS2 instead.")
    intercepts = g[:-1] - slopes * L[:-1]
    return slopes, intercepts


def build_model(
    data: SchedulingData,
    design: PlantDesign,
    par: Optional[OperatingParams] = None,
):
    """Build the GAMSPy MILP and its LP relaxation.

    Returns
    -------
    container, model, relaxed_model, symbols
    """

    if par is None:
        par = OperatingParams()
    if abs(par.pem_curve[0][0] - par.f_min_PEM) > 1e-9:
        raise ValueError("First PEM curve point must equal f_min_PEM.")

    n_t = len(data.time_steps)
    dt = par.dt_h
    slopes, intercepts = pem_curve_segments(par.pem_curve)

    m = gp.Container()

    # ---------------------------------------------------------------------
    # Sets
    # ---------------------------------------------------------------------
    t = gp.Set(m, "t", records=data.time_steps, description="Scheduling time steps")
    k = gp.Set(m, "k", records=[f"k{i}" for i in range(len(slopes))], description="PEM curve segments")

    # ---------------------------------------------------------------------
    # Indexed parameters (updated between scenarios / rolling-horizon steps)
    # ---------------------------------------------------------------------
    cf = gp.Parameter(m, "cf", domain=t, records=data.cf_wind, description="Wind capacity factor [-]")
    price = gp.Parameter(m, "price", domain=t, records=data.price, description="Electricity price [USD/MWh]")
    eps = gp.Parameter(m, "eps_g", domain=t, records=data.eps_grid, description="Grid emission factor [tCO2/MWh]")
    first = gp.Parameter(m, "first", domain=t, records=[(data.time_steps[0], 1.0)])
    last = gp.Parameter(m, "last", domain=t, records=[(data.time_steps[-1], 1.0)])
    slope = gp.Parameter(m, "slope", domain=k, records=slopes, description="PEM curve slope [kg/h per MW]")
    icpt = gp.Parameter(m, "icpt", domain=k, records=intercepts, description="PEM curve intercept [kg/h per MW]")

    # ---------------------------------------------------------------------
    # Design parameters
    # ---------------------------------------------------------------------
    C_wind = gp.Parameter(m, "C_wind", records=design.C_wind_MW, description="Wind capacity [MW]")
    C_PEM = gp.Parameter(m, "C_PEM", records=design.C_PEM_MW, description="PEM capacity [MW]")
    C_BESS = gp.Parameter(m, "C_BESS", records=design.C_BESS_MWh, description="BESS capacity [MWh]")
    C_H2 = gp.Parameter(m, "C_H2", records=design.C_H2_kg, description="H2 tank capacity [kg]")
    m_nom = gp.Parameter(m, "m_nom", records=design.m_MeOH_nom_kgph, description="MeOH nameplate [kg/h]")
    G_max = gp.Parameter(m, "G_max", records=design.G_max_MW, description="Grid connection [MW]")

    # ---------------------------------------------------------------------
    # Initial conditions (updated in rolling-horizon use)
    # ---------------------------------------------------------------------
    soc0 = gp.Parameter(m, "soc0", records=par.soc0)
    h0 = gp.Parameter(m, "h0", records=par.h0)
    u0_P = gp.Parameter(m, "u0_P", records=par.u0_PEM)
    s0_P = gp.Parameter(m, "s0_P", records=par.s0_PEM)
    u0_M = gp.Parameter(m, "u0_M", records=par.u0_MeOH)
    m0_M = gp.Parameter(m, "m0_M", records=par.m0_MeOH_frac * design.m_MeOH_nom_kgph)

    # ---------------------------------------------------------------------
    # Dispatch variables
    # ---------------------------------------------------------------------
    P_imp = gp.Variable(m, "P_imp", domain=t, type="Positive", description="Grid import [MW]")
    P_exp = gp.Variable(m, "P_exp", domain=t, type="Positive", description="Grid export [MW]")
    P_curt = gp.Variable(m, "P_curt", domain=t, type="Positive", description="Wind curtailment [MW]")
    P_ch = gp.Variable(m, "P_ch", domain=t, type="Positive", description="BESS charge [MW]")
    P_dis = gp.Variable(m, "P_dis", domain=t, type="Positive", description="BESS discharge [MW]")
    E = gp.Variable(m, "E", domain=t, type="Positive", description="BESS energy [MWh]")
    P_PEM = gp.Variable(m, "P_PEM", domain=t, type="Positive", description="PEM power when on [MW]")
    P_comp = gp.Variable(m, "P_comp", domain=t, type="Positive", description="H2 compression [MW]")
    m_H2 = gp.Variable(m, "m_H2", domain=t, type="Positive", description="H2 production [kg/h]")
    h_in = gp.Variable(m, "h_in", domain=t, type="Positive", description="H2 into tank [kg/h]")
    h_out = gp.Variable(m, "h_out", domain=t, type="Positive", description="H2 from tank [kg/h]")
    h_sell = gp.Variable(m, "h_sell", domain=t, type="Positive", description="H2 sold [kg/h]")
    H = gp.Variable(m, "H", domain=t, type="Positive", description="H2 inventory [kg]")
    m_M = gp.Variable(m, "m_M", domain=t, type="Positive", description="MeOH production [kg/h]")
    short = gp.Variable(m, "short", type="Positive", description="MeOH shortfall over horizon [t]")

    u_P = gp.Variable(m, "u_P", domain=t, type="Binary", description="PEM on")
    s_P = gp.Variable(m, "s_P", domain=t, type="Binary", description="PEM warm standby")
    y_P = gp.Variable(m, "y_P", domain=t, type="Positive", description="PEM cold start")
    y_P.up[t] = 1.0
    u_M = gp.Variable(m, "u_M", domain=t, type="Binary", description="MeOH synthesis on")
    y_M = gp.Variable(m, "y_M", domain=t, type="Binary", description="MeOH start-up")
    z_M = gp.Variable(m, "z_M", domain=t, type="Binary", description="MeOH shut-down")

    # ---------------------------------------------------------------------
    # Equations
    # ---------------------------------------------------------------------
    # Electricity bus and grid connection.
    eq_power = gp.Equation(m, "eq_power", domain=t)
    eq_power[t] = cf[t] * C_wind - P_curt[t] + P_imp[t] + P_dis[t] == (
        P_exp[t]
        + P_ch[t]
        + P_PEM[t]
        + par.f_sb_PEM * C_PEM * s_P[t]
        + P_comp[t]
        + par.SP_MeOH / 1000.0 * m_M[t]
    )
    eq_curt = gp.Equation(m, "eq_curt", domain=t)
    eq_curt[t] = P_curt[t] <= cf[t] * C_wind
    eq_imp_max = gp.Equation(m, "eq_imp_max", domain=t)
    eq_imp_max[t] = P_imp[t] <= G_max
    eq_exp_max = gp.Equation(m, "eq_exp_max", domain=t)
    eq_exp_max[t] = P_exp[t] <= G_max

    # BESS.
    eq_soc = gp.Equation(m, "eq_soc", domain=t)
    eq_soc[t] = E[t] == (1.0 - par.loss_BESS) * (E[t.lag(1)] + soc0 * C_BESS * first[t]) + (
        par.eta_ch * P_ch[t] - P_dis[t] / par.eta_dis
    ) * dt
    eq_soc_min = gp.Equation(m, "eq_soc_min", domain=t)
    eq_soc_min[t] = E[t] >= par.soc_min * C_BESS
    eq_soc_max = gp.Equation(m, "eq_soc_max", domain=t)
    eq_soc_max[t] = E[t] <= par.soc_max * C_BESS
    if par.bess_exclusive:
        z_B = gp.Variable(m, "z_B", domain=t, type="Binary", description="BESS charging mode")
        eq_ch_max = gp.Equation(m, "eq_ch_max", domain=t)
        eq_ch_max[t] = P_ch[t] <= par.c_rate * C_BESS * z_B[t]
        eq_dis_max = gp.Equation(m, "eq_dis_max", domain=t)
        eq_dis_max[t] = P_dis[t] <= par.c_rate * C_BESS * (1 - z_B[t])
    else:
        eq_ch_max = gp.Equation(m, "eq_ch_max", domain=t)
        eq_ch_max[t] = P_ch[t] <= par.c_rate * C_BESS
        eq_dis_max = gp.Equation(m, "eq_dis_max", domain=t)
        eq_dis_max[t] = P_dis[t] <= par.c_rate * C_BESS

    # PEM electrolyzer: on / standby / off, min load, cold start, part-load curve.
    eq_pem_state = gp.Equation(m, "eq_pem_state", domain=t)
    eq_pem_state[t] = u_P[t] + s_P[t] <= 1
    eq_pem_min = gp.Equation(m, "eq_pem_min", domain=t)
    eq_pem_min[t] = P_PEM[t] >= par.f_min_PEM * C_PEM * u_P[t]
    eq_pem_max = gp.Equation(m, "eq_pem_max", domain=t)
    eq_pem_max[t] = P_PEM[t] <= C_PEM * u_P[t]
    eq_pem_cold = gp.Equation(m, "eq_pem_cold", domain=t)
    eq_pem_cold[t] = y_P[t] >= u_P[t] - (u_P[t.lag(1)] + u0_P * first[t]) - (s_P[t.lag(1)] + s0_P * first[t])
    eq_pem_curve = gp.Equation(m, "eq_pem_curve", domain=[k, t])
    eq_pem_curve[k, t] = m_H2[t] <= slope[k] * P_PEM[t] + icpt[k] * C_PEM * u_P[t]

    # H2 node, tank and compression.
    eq_h2_node = gp.Equation(m, "eq_h2_node", domain=t)
    eq_h2_node[t] = m_H2[t] + h_out[t] == par.gamma_H2 * m_M[t] + h_in[t] + h_sell[t]
    eq_tank = gp.Equation(m, "eq_tank", domain=t)
    eq_tank[t] = H[t] == H[t.lag(1)] + h0 * C_H2 * first[t] + (h_in[t] - h_out[t]) * dt
    eq_tank_min = gp.Equation(m, "eq_tank_min", domain=t)
    eq_tank_min[t] = H[t] >= par.f_min_H2 * C_H2
    eq_tank_max = gp.Equation(m, "eq_tank_max", domain=t)
    eq_tank_max[t] = H[t] <= C_H2
    eq_comp = gp.Equation(m, "eq_comp", domain=t)
    eq_comp[t] = P_comp[t] == par.SP_comp / 1000.0 * h_in[t]
    eq_sell_max = gp.Equation(m, "eq_sell_max", domain=t)
    eq_sell_max[t] = h_sell[t] <= par.H2_sale_max_kgph

    # Methanol synthesis: on/off, min load, ramp, start/stop logic, min up/down.
    eq_m_min = gp.Equation(m, "eq_m_min", domain=t)
    eq_m_min[t] = m_M[t] >= par.f_min_MeOH * m_nom * u_M[t]
    eq_m_max = gp.Equation(m, "eq_m_max", domain=t)
    eq_m_max[t] = m_M[t] <= m_nom * u_M[t]
    eq_m_logic = gp.Equation(m, "eq_m_logic", domain=t)
    eq_m_logic[t] = u_M[t] - (u_M[t.lag(1)] + u0_M * first[t]) == y_M[t] - z_M[t]
    eq_m_yz = gp.Equation(m, "eq_m_yz", domain=t)
    eq_m_yz[t] = y_M[t] + z_M[t] <= 1
    eq_m_ramp_up = gp.Equation(m, "eq_m_ramp_up", domain=t)
    eq_m_ramp_up[t] = m_M[t] - (m_M[t.lag(1)] + m0_M * first[t]) <= (
        par.ramp_MeOH * m_nom * dt + par.f_min_MeOH * m_nom * y_M[t]
    )
    eq_m_ramp_dn = gp.Equation(m, "eq_m_ramp_dn", domain=t)
    eq_m_ramp_dn[t] = (m_M[t.lag(1)] + m0_M * first[t]) - m_M[t] <= (
        par.ramp_MeOH * m_nom * dt + par.f_min_MeOH * m_nom * z_M[t]
    )
    # Minimum up/down time; the status history before t=0 is assumed feasible.
    eq_m_minup = gp.Equation(m, "eq_m_minup", domain=t)
    eq_m_minup[t] = functools.reduce(operator.add, [y_M[t]] + [y_M[t.lag(j)] for j in range(1, par.UT_MeOH)]) <= u_M[t]
    eq_m_mindn = gp.Equation(m, "eq_m_mindn", domain=t)
    eq_m_mindn[t] = functools.reduce(operator.add, [z_M[t]] + [z_M[t.lag(j)] for j in range(1, par.DT_MeOH)]) <= 1 - u_M[t]

    # Production target over the horizon (soft).
    target_t = par.target_frac_MeOH * design.m_MeOH_nom_kgph * n_t * dt / 1000.0
    eq_target = gp.Equation(m, "eq_target")
    eq_target[...] = gp.Sum(t, m_M[t] * dt) / 1000.0 + short >= target_t

    # Carbon-intensity cap on grid imports per tonne of methanol.
    if par.CI_max is not None:
        eq_ci = gp.Equation(m, "eq_ci")
        eq_ci[...] = gp.Sum(t, eps[t] * P_imp[t] * dt) <= par.CI_max * gp.Sum(t, m_M[t] * dt) / 1000.0

    # Terminal storage conditions.
    if par.terminal_storage:
        eq_term_E = gp.Equation(m, "eq_term_E")
        eq_term_E[...] = gp.Sum(t, last[t] * E[t]) >= soc0 * C_BESS
        eq_term_H = gp.Equation(m, "eq_term_H")
        eq_term_H[...] = gp.Sum(t, last[t] * H[t]) >= h0 * C_H2

    # ---------------------------------------------------------------------
    # Objective [USD over the horizon]
    # ---------------------------------------------------------------------
    cost_grid = gp.Sum(
        t, ((price[t] + par.grid_fee + par.tax_CO2 * eps[t]) * P_imp[t] - price[t] * P_exp[t]) * dt
    )
    cost_var = gp.Sum(
        t,
        (
            par.c_water_H2 * m_H2[t]
            + par.c_tiebreak_PEM * P_PEM[t]
            + (par.c_CO2_feed * par.gamma_CO2 - par.p_MeOH) / 1000.0 * m_M[t]
            - par.p_H2_sale * h_sell[t]
        )
        * dt,
    )
    cost_start = par.c_cold_start_PEM * C_PEM * gp.Sum(t, y_P[t]) + par.c_start_MeOH * gp.Sum(t, y_M[t])
    total_cost = cost_grid + cost_var + cost_start + par.c_short_MeOH * short

    model = gp.Model(
        m,
        name="ptm_scheduling",
        equations=m.getEquations(),
        problem="MIP",
        sense=gp.Sense.MIN,
        objective=total_cost,
    )
    relaxed = gp.Model(
        m,
        name="ptm_scheduling_rmip",
        equations=m.getEquations(),
        problem="RMIP",
        sense=gp.Sense.MIN,
        objective=total_cost,
    )

    symbols = {
        "sets": {"t": t, "k": k},
        "parameters": {"cf": cf, "price": price, "eps": eps, "soc0": soc0, "h0": h0},
        "variables": {
            "P_imp": P_imp,
            "P_exp": P_exp,
            "P_curt": P_curt,
            "P_ch": P_ch,
            "P_dis": P_dis,
            "E": E,
            "P_PEM": P_PEM,
            "P_comp": P_comp,
            "m_H2": m_H2,
            "h_in": h_in,
            "h_out": h_out,
            "h_sell": h_sell,
            "H": H,
            "m_M": m_M,
            "u_P": u_P,
            "s_P": s_P,
            "y_P": y_P,
            "u_M": u_M,
            "y_M": y_M,
            "z_M": z_M,
        },
        "short": short,
    }
    return m, model, relaxed, symbols


def solve(model, solver: str = "cplex", gap: float = 0.005, time_limit: float = 600.0, threads: int = 0, output=None):
    """Solve and return (wall-clock time [s], GAMSPy solve summary)."""

    t0 = time.perf_counter()
    summary = model.solve(
        solver=solver,
        options=gp.Options(relative_optimality_gap=gap, time_limit=time_limit, threads=threads),
        output=output,
    )
    return time.perf_counter() - t0, summary


def hourly_results(data: SchedulingData, design: PlantDesign, par: OperatingParams, symbols) -> pd.DataFrame:
    """Hourly levels of all dispatch variables."""

    df = pd.DataFrame({"t": data.time_steps, "cf": data.cf_wind, "price": data.price, "eps": data.eps_grid})
    for name, var in symbols["variables"].items():
        dense = var.toDense()
        df[name] = np.zeros(len(df)) if dense is None else np.asarray(dense, dtype=float)
    df["P_wind"] = df["cf"] * design.C_wind_MW
    df["P_sb"] = par.f_sb_PEM * design.C_PEM_MW * df["s_P"]
    df["P_MeOH"] = par.SP_MeOH / 1000.0 * df["m_M"]
    return df


def kpis(df: pd.DataFrame, design: PlantDesign, par: OperatingParams) -> dict:
    """Horizon totals from the hourly results."""

    dt = par.dt_h
    meoh_t = df["m_M"].sum() * dt / 1000.0
    emis_t = (df["eps"] * df["P_imp"]).sum() * dt
    return {
        "MeOH_t": meoh_t,
        "MeOH_load_factor": meoh_t / (design.m_MeOH_nom_kgph * len(df) * dt / 1000.0),
        "H2_prod_t": df["m_H2"].sum() * dt / 1000.0,
        "H2_sold_t": df["h_sell"].sum() * dt / 1000.0,
        "import_MWh": df["P_imp"].sum() * dt,
        "export_MWh": df["P_exp"].sum() * dt,
        "curtail_MWh": df["P_curt"].sum() * dt,
        "wind_MWh": df["P_wind"].sum() * dt,
        "emissions_tCO2": emis_t,
        "CI_tCO2_per_tMeOH": emis_t / meoh_t if meoh_t > 0 else float("nan"),
        "PEM_on_h": df["u_P"].sum() * dt,
        "PEM_standby_h": df["s_P"].sum() * dt,
        "PEM_cold_starts": df["y_P"].sum(),
        "MeOH_on_h": df["u_M"].sum() * dt,
        "MeOH_starts": df["y_M"].sum(),
        "grid_cost_USD": (df["price"] * (df["P_imp"] - df["P_exp"])).sum() * dt,
    }

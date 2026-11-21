"""
GAMSPy MILP for hourly scheduling of a wind- and PV-powered power-to-methanol
(PtM) plant with bidirectional grid exchange, a battery (BESS) and a compressed
H2 tank.

Process boundary:
    wind + PV + grid -> PEM electrolyzer -> (H2 tank) -> methanol synthesis
    BESS on the electricity bus; optional H2 sales to an external market
    distillation="explicit": synthesis -> crude methanol tank -> FVC column,
    with a pressurized-water thermal store fed by the synthesis heat

Changes relative to the MasCOR operational model (Kim et al., arXiv:2603.03484,
mascor/solvers/GLOBAL_solver.py):
    1. PEM electrolyzer has on / warm-standby / off states, a minimum load,
       standby power and a cold-start cost. The load-dependent H2 curve is
       concave and is represented by its piecewise-linear upper envelope, so it
       needs no extra binaries.
    2. Methanol synthesis stays on and modulates between a minimum load and
       nameplate with ramp limits (as in Mucci et al., 2023). Nameplate may
       exceed the production requirement (oversizing). An on/off
       unit-commitment variant is kept for sensitivity runs.
    3. PV generation and explicit curtailment, so the grid-direction binaries
       of MasCOR are not needed.
    4. Hourly grid emission factor and a carbon-intensity cap on methanol.
       CI_max = 1.435802 tCO2/tMeOH reproduces MasCOR's "net-negative" rule.
       No carbon tax on grid imports by default: power generation is covered
       by the EU ETS, whose cost is already in wholesale prices.
    5. A methanol production requirement (soft, with shortfall penalty)
       replaces unlimited H2 sales as the reason to run the plant.
    6. Terminal storage conditions remove end-of-horizon depletion.

Optional extensions (all off by default):
    7. distillation="explicit": crude methanol tank, distillation column with
       flash vapor circulation (FVC) and a pressurized-water thermal store
       (TES) that also receives the synthesis reaction heat; optional electric
       boiler. Column and FVC data are scaled from Qi et al. (2025, AIChE J
       71:e18750). The FVC recovers all condenser heat when on, so its power
       is linear in the heat sent to storage (marginal COP = 1).
    8. Synthesis part-load map (syn_curve), load-dependent ramp limits and
       H2 losses in load changes; placeholders until calibrated with Aspen
       Dynamics.
    9. Green-power matching windows: grid import as a share of plant load,
       and export plus curtailment as a share of available generation.

Parameter sources are documented in docs/MILP_model.md.

Units used in this model:
    time: h
    power: MW, electrical energy: MWh, heat: MW_th and MWh_th
    H2 and methanol flows: kg/h, H2 and crude methanol inventories: kg
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


def linear_sec_curve(sec_full=55.7, sec_zero=39.4, loads=(0.10, 0.25, 0.50, 0.75, 1.00)) -> tuple:
    """Electrolyzer specific energy consumption of Chen & Yang (2021), Eq. 20.

    SEC(L) = sec_full - (1 - L) * (sec_full - sec_zero), kWh/kg H2, evaluated
    at the given load fractions L.
    """

    return tuple((L, sec_full - (1.0 - L) * (sec_full - sec_zero)) for L in loads)


# Default curve: linear SEC from Chen & Yang (2021), 55.7 kWh/kg at full load.
PEM_CURVE_CHEN_YANG = linear_sec_curve()
# Earlier placeholder with a system-efficiency peak near 20 % load; sensitivity only.
PEM_CURVE_BOP_PEAK = ((0.10, 52.5), (0.20, 50.0), (0.50, 52.0), (1.00, 55.7))


@dataclass
class PlantDesign:
    """Installed capacities and production requirement."""

    C_wind_MW: float = 50.0
    C_PV_MW: float = 0.0
    C_PEM_MW: float = 20.0
    C_BESS_MWh: float = 20.0
    C_H2_kg: float = 1000.0
    m_MeOH_nom_kgph: float = 1000.0  # synthesis nameplate capacity
    D_MeOH_kgph: Optional[float] = None  # average production requirement; None -> nameplate
    G_max_MW: float = 50.0
    # Explicit downstream section (distillation="explicit" only)
    C_crude_kg: float = 24000.0  # crude methanol tank, as methanol content [kg]
    E_TES_MWh: float = 8.0  # pressurized-water thermal store [MWh_th]
    W_FVC_MW: float = 0.40  # FVC compressor capacity [MW]; 0 -> no FVC
    C_eboiler_MW: float = 0.0  # electric boiler charging the thermal store [MW]

    @property
    def demand_kgph(self) -> float:
        return self.m_MeOH_nom_kgph if self.D_MeOH_kgph is None else self.D_MeOH_kgph

    @property
    def oversize(self) -> float:
        """Nameplate / requirement (kappa)."""
        return self.m_MeOH_nom_kgph / self.demand_kgph


@dataclass
class OperatingParams:
    """Technical and economic parameters.

    Values marked ASSUMPTION are placeholders that still need plant- or
    literature-specific calibration.
    """

    dt_h: float = 1.0

    # PEM electrolyzer
    f_min_PEM: float = 0.10  # minimum load fraction
    f_sb_PEM: float = 0.02  # warm-standby power fraction (literature: 1-5 %)
    pem_curve: tuple = PEM_CURVE_CHEN_YANG  # (load fraction, system kWh/kg H2)
    c_cold_start_PEM: float = 51.5  # USD per MW installed per cold start (50 EUR/MW)
    # Tie-breaker: at zero price, running the stack below its H2 curve (i.e.
    # wasting power) is otherwise as cheap as exporting or curtailing.
    c_tiebreak_PEM: float = 0.01  # USD/MWh

    # Methanol synthesis
    f_min_MeOH: float = 0.20  # minimum load fraction of nameplate
    ramp_MeOH: float = 0.50  # fraction of nameplate per h
    SP_MeOH: float = 0.65702  # kWh/kg MeOH (MasCOR); loop electricity only in explicit mode. ASSUMPTION
    gamma_H2: float = 0.19576  # kg H2/kg MeOH (MasCOR)
    gamma_CO2: float = 1.435802  # kg CO2/kg MeOH (MasCOR)
    # On/off variant for sensitivity runs only; its parameters are ASSUMPTION.
    methanol_uc: bool = False
    UT_MeOH: int = 8  # minimum up time [h]. ASSUMPTION
    DT_MeOH: int = 4  # minimum down time [h]. ASSUMPTION
    c_start_MeOH: float = 5000.0  # USD per start. ASSUMPTION
    # Synthesis dynamics at the hourly scale, to be calibrated with Aspen Dynamics.
    # Part-load map ((load fraction, kg H2/kg MeOH, kWh/kg MeOH), ...); None ->
    # constant gamma_H2 and SP_MeOH. Total H2 use and loop power must be convex in load.
    syn_curve: Optional[tuple] = None
    ramp_MeOH_R1: float = 0.0  # load-dependent ramp term [1/h]: limit = (ramp_MeOH*m_nom + R1*m_prev)*dt
    alpha_up_H2: float = 0.0  # kg H2 lost per kg/h of upward load change. ASSUMPTION
    alpha_dn_H2: float = 0.0  # kg H2 lost per kg/h of downward load change. ASSUMPTION

    # H2 compression and storage
    SP_comp: float = 3.03  # kWh/kg H2 compressed into the tank (Chen & Yang, 2021)
    f_min_H2: float = 0.0  # minimum tank fraction (cushion gas)

    # BESS (MasCOR values)
    eta_ch: float = 0.95
    eta_dis: float = 0.95
    loss_BESS: float = 0.05 / 720.0  # self-discharge per h
    soc_min: float = 0.10
    soc_max: float = 0.90
    c_rate: float = 0.30  # power / energy [1/h]
    bess_exclusive: bool = False  # add charge/discharge binary

    # Downstream section: "lumped" keeps distillation inside SP_MeOH and the
    # requirement on synthesis output; "explicit" adds the crude tank, the FVC
    # column and the thermal store.
    distillation: str = "lumped"
    beta_dist: float = 0.0  # column throughput band around D [-]; 0 -> constant column load
    ramp_dist: Optional[float] = None  # column throughput ramp [fraction of D per h]; None -> band only
    # Column and FVC per kg MeOH distilled, scaled from Qi et al. (2025): reboiler
    # 1167 kW, condenser 1071 kW, 55 kmol/h (1762 kg/h) methanol in the distillate.
    q_reb: float = 0.6623  # kWh_th/kg
    q_cond: float = 0.6078  # kWh_th/kg
    q_str_min: float = 0.1078  # kWh_th/kg FVC heat to storage at its minimum point (190 kW)
    q_str_max: float = 0.7422  # kWh_th/kg FVC heat to storage at its two-stage (a) maximum (1308 kW)
    # Synthesis heat recovered into the thermal store; the full reaction enthalpy
    # (CO2 + 3 H2 -> CH3OH + H2O, -49.5 kJ/mol) is 0.43 kWh_th/kg.
    q_syn: float = 0.30  # kWh_th/kg MeOH synthesized. ASSUMPTION
    loss_TES: float = 0.001  # thermal store loss per h. ASSUMPTION (Qi et al. assumed none)
    eta_eboiler: float = 0.99
    c_cooling: float = 1.2744  # USD/MWh_th cooling water (0.354 USD/GJ, Qi et al. 2025)

    # Green-power matching windows (None disables)
    match_window_h: Optional[int] = None  # window length [h], e.g. 8760, 720, 1
    phi_import: Optional[float] = None  # grid import <= phi_import * plant load in each window
    phi_spill: Optional[float] = None  # export + curtailment <= phi_spill * available generation

    # Economics
    grid_fee: float = 0.0  # USD/MWh added to the import price
    tax_CO2: float = 0.0  # USD/tCO2 on import emissions; MasCOR used 28.10 for Denmark
    # Tie-breaker: at zero price, simultaneous import and export is otherwise free
    # and only inflates imports and emissions up to the carbon cap.
    c_tiebreak_grid: float = 0.01  # USD/MWh on import and export
    c_CO2_feed: float = 50.0  # USD/t captured CO2 feed (MasCOR)
    c_water_H2: float = 0.1284  # USD/kg H2 water/consumables (MasCOR material_price)
    p_H2_sale: float = 0.0  # USD/kg; MasCOR uses 5.0
    H2_sale_max_kgph: float = 0.0  # 0 disables H2 sales
    p_MeOH: float = 0.0  # USD/t methanol revenue
    c_short_MeOH: float = 5000.0  # USD/t production shortfall
    CI_max: Optional[float] = 1.435802  # tCO2/tMeOH on grid imports; None disables

    # Initial and terminal conditions
    soc0: float = 0.50  # BESS SOC at t=0 [-]
    h0: float = 0.50  # H2 tank level at t=0 [-]
    u0_PEM: int = 1
    s0_PEM: int = 0
    u0_MeOH: int = 1  # on/off variant only
    m0_MeOH_frac: Optional[float] = None  # initial load of nameplate; None -> requirement
    crude0: float = 0.50  # crude tank level at t=0 [-]
    e0_TES: float = 0.50  # thermal store level at t=0 [-]
    terminal_storage: bool = True  # end levels of all stores >= initial


@dataclass
class SchedulingData:
    """Hourly exogenous inputs."""

    time_steps: list[str]
    cf_wind: np.ndarray  # wind capacity factor [-]
    price: np.ndarray  # electricity price [USD/MWh]
    eps_grid: np.ndarray  # grid emission factor [tCO2/MWh]
    cf_pv: Optional[np.ndarray] = None  # PV capacity factor [-]; None -> zeros

    def __post_init__(self):
        if self.cf_pv is None:
            self.cf_pv = np.zeros(len(self.time_steps))

    @classmethod
    def from_arrays(cls, cf_wind, price, eps_grid=0.5, cf_pv=None) -> "SchedulingData":
        cf_wind = np.asarray(cf_wind, dtype=float)
        n = len(cf_wind)
        eps = np.broadcast_to(np.asarray(eps_grid, dtype=float), (n,)).copy()
        pv = None if cf_pv is None else np.asarray(cf_pv, dtype=float)
        return cls([f"t{i:04d}" for i in range(1, n + 1)], cf_wind, np.asarray(price, dtype=float), eps, pv)


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


def syn_curve_segments(curve, m_nom: float) -> tuple[tuple, tuple]:
    """Secant lines of total synthesis H2 use [kg/h] and loop power [MW] versus output [kg/h].

    Both totals must be convex in output, so the maximum over the secant lines
    equals the piecewise-linear interpolant and
        h2_syn >= a_k * m_M + b_k * u_M   for all k
    is exact without SOS2 binaries.
    """

    L = np.array([p[0] for p in curve], dtype=float)
    out = L * m_nom
    segs = []
    for name, total in (("H2 use", out * np.array([p[1] for p in curve])),
                        ("loop power", out * np.array([p[2] for p in curve]) / 1000.0)):
        slopes = np.diff(total) / np.diff(out)
        if np.any(np.diff(slopes) < -1e-12):
            raise ValueError(f"Synthesis {name} is not convex in load; use SOS2 instead.")
        segs.append((slopes, total[:-1] - slopes * out[:-1]))
    return segs[0], segs[1]


def add_downstream_section(m, t, first, last, design: PlantDesign, par: OperatingParams, m_syn, dt: float):
    """Crude methanol tank, FVC distillation column and thermal store.

    The column runs in the two modes of Qi et al. (2025): with the FVC on
    (d_F = 1) the compressor recovers all condenser heat, covers the reboiler
    and sends surplus heat Q_str to the store; with the FVC off the reboiler
    draws on the store and the condenser uses cooling water. The store also
    receives the synthesis heat and, optionally, an electric boiler.

    Returns (variables, electric load [MW] expression over t, operating cost [USD]).
    """

    D = design.demand_kgph
    m_hi = (1.0 + par.beta_dist) * D

    C_crude = gp.Parameter(m, "C_crude", records=design.C_crude_kg, description="Crude tank [kg MeOH]")
    E_cap = gp.Parameter(m, "E_cap", records=design.E_TES_MWh, description="Thermal store [MWh_th]")
    W_cap = gp.Parameter(m, "W_cap", records=design.W_FVC_MW, description="FVC compressor [MW]")
    C_eb = gp.Parameter(m, "C_eb", records=design.C_eboiler_MW, description="Electric boiler [MW]")
    crude0 = gp.Parameter(m, "crude0", records=par.crude0)
    e0 = gp.Parameter(m, "e0_TES", records=par.e0_TES)

    m_D = gp.Variable(m, "m_D", domain=t, type="Positive", description="Column throughput, MeOH delivered [kg/h]")
    m_D.up[t] = m_hi
    q_D = gp.Variable(m, "q_D", domain=t, type="Positive", description="Offtake shortfall below the band [kg/h]")
    I_c = gp.Variable(m, "I_c", domain=t, type="Positive", description="Crude tank, MeOH content [kg]")
    d_F = gp.Variable(m, "d_F", domain=t, type="Binary", description="FVC on")
    z_F = gp.Variable(m, "z_F", domain=t, type="Positive", description="MeOH distilled with the FVC on [kg/h]")
    Q_str = gp.Variable(m, "Q_str", domain=t, type="Positive", description="FVC heat to the store [MW_th]")
    W_F = gp.Variable(m, "W_F", domain=t, type="Positive", description="FVC compressor power [MW]")
    P_eb = gp.Variable(m, "P_eb", domain=t, type="Positive", description="Electric boiler [MW]")
    Q_dump = gp.Variable(m, "Q_dump", domain=t, type="Positive", description="Heat rejected from the store [MW_th]")
    E_T = gp.Variable(m, "E_T", domain=t, type="Positive", description="Thermal store [MWh_th]")

    # Crude methanol tank between synthesis and column.
    eq_crude = gp.Equation(m, "eq_crude", domain=t)
    eq_crude[t] = I_c[t] == I_c[t.lag(1)] + crude0 * C_crude * first[t] + (m_syn[t] - m_D[t]) * dt
    eq_crude_max = gp.Equation(m, "eq_crude_max", domain=t)
    eq_crude_max[t] = I_c[t] <= C_crude

    # Column throughput band around the offtake rate (soft lower bound).
    eq_dist_lo = gp.Equation(m, "eq_dist_lo", domain=t)
    eq_dist_lo[t] = m_D[t] + q_D[t] >= (1.0 - par.beta_dist) * D
    if par.ramp_dist is not None:
        m_D_prev = m_D[t.lag(1)] + D * first[t]
        eq_dist_up = gp.Equation(m, "eq_dist_up", domain=t)
        eq_dist_up[t] = m_D[t] - m_D_prev <= par.ramp_dist * D * dt
        eq_dist_dn = gp.Equation(m, "eq_dist_dn", domain=t)
        eq_dist_dn[t] = m_D_prev - m_D[t] <= par.ramp_dist * D * dt

    # z_F = m_D * d_F, exact for binary d_F.
    eq_z_on = gp.Equation(m, "eq_z_on", domain=t)
    eq_z_on[t] = z_F[t] <= m_hi * d_F[t]
    eq_z_m = gp.Equation(m, "eq_z_m", domain=t)
    eq_z_m[t] = z_F[t] <= m_D[t]
    eq_z_lo = gp.Equation(m, "eq_z_lo", domain=t)
    eq_z_lo[t] = z_F[t] >= m_D[t] - m_hi * (1 - d_F[t])

    # FVC: energy balance Q_reb + Q_str = Q_cond + W when on, i.e. marginal COP = 1.
    eq_fvc_power = gp.Equation(m, "eq_fvc_power", domain=t)
    eq_fvc_power[t] = W_F[t] == Q_str[t] + (par.q_reb - par.q_cond) / 1000.0 * z_F[t]
    eq_fvc_min = gp.Equation(m, "eq_fvc_min", domain=t)
    eq_fvc_min[t] = Q_str[t] >= par.q_str_min / 1000.0 * z_F[t]
    eq_fvc_max = gp.Equation(m, "eq_fvc_max", domain=t)
    eq_fvc_max[t] = Q_str[t] <= par.q_str_max / 1000.0 * z_F[t]
    eq_fvc_cap = gp.Equation(m, "eq_fvc_cap", domain=t)
    eq_fvc_cap[t] = W_F[t] <= W_cap
    eq_eb_cap = gp.Equation(m, "eq_eb_cap", domain=t)
    eq_eb_cap[t] = P_eb[t] <= C_eb

    # Thermal store with fixed hot and cold temperatures (two-tank, linear in stored mass).
    # The reboiler draws on it whenever the FVC is off.
    eq_tes = gp.Equation(m, "eq_tes", domain=t)
    eq_tes[t] = E_T[t] == (1.0 - par.loss_TES) * (E_T[t.lag(1)] + e0 * E_cap * first[t]) + (
        Q_str[t]
        + par.q_syn / 1000.0 * m_syn[t]
        + par.eta_eboiler * P_eb[t]
        - par.q_reb / 1000.0 * (m_D[t] - z_F[t])
        - Q_dump[t]
    ) * dt
    eq_tes_max = gp.Equation(m, "eq_tes_max", domain=t)
    eq_tes_max[t] = E_T[t] <= E_cap

    if par.terminal_storage:
        eq_term_I = gp.Equation(m, "eq_term_I")
        eq_term_I[...] = gp.Sum(t, last[t] * I_c[t]) >= crude0 * C_crude
        eq_term_T = gp.Equation(m, "eq_term_T")
        eq_term_T[...] = gp.Sum(t, last[t] * E_T[t]) >= e0 * E_cap

    # Cooling water for the condenser with the FVC off and for rejected heat.
    cost = gp.Sum(t, par.c_cooling * (par.q_cond / 1000.0 * (m_D[t] - z_F[t]) + Q_dump[t]) * dt)
    load = W_F[t] + P_eb[t]
    variables = {
        "m_D": m_D, "q_D": q_D, "I_c": I_c, "d_F": d_F, "z_F": z_F,
        "Q_str": Q_str, "W_F": W_F, "P_eb": P_eb, "Q_dump": Q_dump, "E_T": E_T,
    }
    return variables, load, cost


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
    if par.distillation not in ("lumped", "explicit"):
        raise ValueError('distillation must be "lumped" or "explicit".')
    if par.syn_curve is not None and abs(par.syn_curve[0][0] - par.f_min_MeOH) > 1e-9:
        raise ValueError("First synthesis curve point must equal f_min_MeOH.")
    explicit = par.distillation == "explicit"

    n_t = len(data.time_steps)
    dt = par.dt_h
    slopes, intercepts = pem_curve_segments(par.pem_curve)
    m0_frac = 1.0 / design.oversize if par.m0_MeOH_frac is None else par.m0_MeOH_frac

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
    cf_pv = gp.Parameter(m, "cf_pv", domain=t, records=data.cf_pv, description="PV capacity factor [-]")
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
    C_pv = gp.Parameter(m, "C_pv", records=design.C_PV_MW, description="PV capacity [MW]")
    C_PEM = gp.Parameter(m, "C_PEM", records=design.C_PEM_MW, description="PEM capacity [MW]")
    C_BESS = gp.Parameter(m, "C_BESS", records=design.C_BESS_MWh, description="BESS capacity [MWh]")
    C_H2 = gp.Parameter(m, "C_H2", records=design.C_H2_kg, description="H2 tank capacity [kg]")
    m_nom = gp.Parameter(m, "m_nom", records=design.m_MeOH_nom_kgph, description="MeOH nameplate [kg/h]")
    D_req = gp.Parameter(m, "D_req", records=design.demand_kgph, description="MeOH requirement [kg/h]")
    G_max = gp.Parameter(m, "G_max", records=design.G_max_MW, description="Grid connection [MW]")

    # ---------------------------------------------------------------------
    # Initial conditions (updated in rolling-horizon use)
    # ---------------------------------------------------------------------
    soc0 = gp.Parameter(m, "soc0", records=par.soc0)
    h0 = gp.Parameter(m, "h0", records=par.h0)
    u0_P = gp.Parameter(m, "u0_P", records=par.u0_PEM)
    s0_P = gp.Parameter(m, "s0_P", records=par.s0_PEM)
    m0_M = gp.Parameter(m, "m0_M", records=m0_frac * design.m_MeOH_nom_kgph)

    # ---------------------------------------------------------------------
    # Dispatch variables
    # ---------------------------------------------------------------------
    P_imp = gp.Variable(m, "P_imp", domain=t, type="Positive", description="Grid import [MW]")
    P_exp = gp.Variable(m, "P_exp", domain=t, type="Positive", description="Grid export [MW]")
    P_curt = gp.Variable(m, "P_curt", domain=t, type="Positive", description="Wind and PV curtailment [MW]")
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

    # ---------------------------------------------------------------------
    # Equations
    # ---------------------------------------------------------------------
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

    # Methanol synthesis.
    extra_vars = {}
    eq_m_min = gp.Equation(m, "eq_m_min", domain=t)
    eq_m_max = gp.Equation(m, "eq_m_max", domain=t)
    eq_m_ramp_up = gp.Equation(m, "eq_m_ramp_up", domain=t)
    eq_m_ramp_dn = gp.Equation(m, "eq_m_ramp_dn", domain=t)
    m_prev = m_M[t.lag(1)] + m0_M * first[t]
    ramp_lim = par.ramp_MeOH * m_nom * dt
    if par.ramp_MeOH_R1:
        # Load-dependent ramp limit, e.g. slower changes at low load.
        ramp_lim = ramp_lim + par.ramp_MeOH_R1 * m_prev * dt
    if par.methanol_uc:
        # On/off variant: start/stop logic, min load when on, min up/down time.
        u0_M = gp.Parameter(m, "u0_M", records=par.u0_MeOH)
        u_M = gp.Variable(m, "u_M", domain=t, type="Binary", description="MeOH synthesis on")
        y_M = gp.Variable(m, "y_M", domain=t, type="Binary", description="MeOH start-up")
        z_M = gp.Variable(m, "z_M", domain=t, type="Binary", description="MeOH shut-down")
        extra_vars = {"u_M": u_M, "y_M": y_M, "z_M": z_M}
        eq_m_min[t] = m_M[t] >= par.f_min_MeOH * m_nom * u_M[t]
        eq_m_max[t] = m_M[t] <= m_nom * u_M[t]
        eq_m_logic = gp.Equation(m, "eq_m_logic", domain=t)
        eq_m_logic[t] = u_M[t] - (u_M[t.lag(1)] + u0_M * first[t]) == y_M[t] - z_M[t]
        eq_m_yz = gp.Equation(m, "eq_m_yz", domain=t)
        eq_m_yz[t] = y_M[t] + z_M[t] <= 1
        eq_m_ramp_up[t] = m_M[t] - m_prev <= ramp_lim + par.f_min_MeOH * m_nom * y_M[t]
        eq_m_ramp_dn[t] = m_prev - m_M[t] <= ramp_lim + par.f_min_MeOH * m_nom * z_M[t]
        # Minimum up/down time; the status history before t=0 is assumed feasible.
        eq_m_minup = gp.Equation(m, "eq_m_minup", domain=t)
        eq_m_minup[t] = functools.reduce(operator.add, [y_M[t]] + [y_M[t.lag(j)] for j in range(1, par.UT_MeOH)]) <= u_M[t]
        eq_m_mindn = gp.Equation(m, "eq_m_mindn", domain=t)
        eq_m_mindn[t] = functools.reduce(operator.add, [z_M[t]] + [z_M[t.lag(j)] for j in range(1, par.DT_MeOH)]) <= 1 - u_M[t]
        cost_start_M = par.c_start_MeOH * gp.Sum(t, y_M[t])
        on_M = u_M[t]
    else:
        # Base case: always on, modulating between minimum load and nameplate.
        eq_m_min[t] = m_M[t] >= par.f_min_MeOH * m_nom
        eq_m_max[t] = m_M[t] <= m_nom
        eq_m_ramp_up[t] = m_M[t] - m_prev <= ramp_lim
        eq_m_ramp_dn[t] = m_prev - m_M[t] <= ramp_lim
        cost_start_M = 0
        on_M = 1

    # Synthesis H2 use and loop power: constant per kg, or the part-load map.
    if par.syn_curve is None:
        h2_syn = par.gamma_H2 * m_M[t]
        P_syn = par.SP_MeOH / 1000.0 * m_M[t]
    else:
        (a_h, b_h), (a_p, b_p) = syn_curve_segments(par.syn_curve, design.m_MeOH_nom_kgph)
        ks = gp.Set(m, "ks", records=[f"s{i}" for i in range(len(a_h))], description="Synthesis curve segments")
        a_H = gp.Parameter(m, "a_H", domain=ks, records=a_h)
        b_H = gp.Parameter(m, "b_H", domain=ks, records=b_h)
        a_P = gp.Parameter(m, "a_P", domain=ks, records=a_p)
        b_P = gp.Parameter(m, "b_P", domain=ks, records=b_p)
        h_S = gp.Variable(m, "h_S", domain=t, type="Positive", description="Synthesis H2 use [kg/h]")
        P_S = gp.Variable(m, "P_S", domain=t, type="Positive", description="Synthesis loop power [MW]")
        eq_syn_h2 = gp.Equation(m, "eq_syn_h2", domain=[ks, t])
        eq_syn_h2[ks, t] = h_S[t] >= a_H[ks] * m_M[t] + b_H[ks] * on_M
        eq_syn_pow = gp.Equation(m, "eq_syn_pow", domain=[ks, t])
        eq_syn_pow[ks, t] = P_S[t] >= a_P[ks] * m_M[t] + b_P[ks] * on_M
        extra_vars.update({"h_S": h_S, "P_S": P_S})
        h2_syn = h_S[t]
        P_syn = P_S[t]
    if par.alpha_up_H2 or par.alpha_dn_H2:
        # H2 lost in load changes (purge, off-spec crude), convex in the change.
        l_H2 = gp.Variable(m, "l_H2", domain=t, type="Positive", description="H2 lost in load changes [kg/h]")
        eq_loss_up = gp.Equation(m, "eq_loss_up", domain=t)
        eq_loss_up[t] = l_H2[t] * dt >= par.alpha_up_H2 * (m_M[t] - m_prev)
        eq_loss_dn = gp.Equation(m, "eq_loss_dn", domain=t)
        eq_loss_dn[t] = l_H2[t] * dt >= par.alpha_dn_H2 * (m_prev - m_M[t])
        extra_vars["l_H2"] = l_H2
        h2_syn = h2_syn + l_H2[t]

    # Downstream: methanol delivered, extra electric load and cost.
    if explicit:
        down_vars, down_load, cost_down = add_downstream_section(m, t, first, last, design, par, m_M, dt)
        extra_vars.update(down_vars)
        m_out = down_vars["m_D"]
    else:
        down_load, cost_down = 0, 0
        m_out = m_M

    # Electricity bus and grid connection.
    load = P_PEM[t] + par.f_sb_PEM * C_PEM * s_P[t] + P_comp[t] + P_syn + down_load
    eq_power = gp.Equation(m, "eq_power", domain=t)
    eq_power[t] = cf[t] * C_wind + cf_pv[t] * C_pv - P_curt[t] + P_imp[t] + P_dis[t] == P_exp[t] + P_ch[t] + load
    eq_curt = gp.Equation(m, "eq_curt", domain=t)
    eq_curt[t] = P_curt[t] <= cf[t] * C_wind + cf_pv[t] * C_pv
    eq_imp_max = gp.Equation(m, "eq_imp_max", domain=t)
    eq_imp_max[t] = P_imp[t] <= G_max
    eq_exp_max = gp.Equation(m, "eq_exp_max", domain=t)
    eq_exp_max[t] = P_exp[t] <= G_max

    # H2 node, tank and compression.
    eq_h2_node = gp.Equation(m, "eq_h2_node", domain=t)
    eq_h2_node[t] = m_H2[t] + h_out[t] == h2_syn + h_in[t] + h_sell[t]
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

    # Production requirement over the horizon (soft). In explicit mode the
    # shortfall is the hourly offtake shortfall below the column band.
    if explicit:
        eq_short = gp.Equation(m, "eq_short")
        eq_short[...] = short == gp.Sum(t, down_vars["q_D"][t] * dt) / 1000.0
    eq_target = gp.Equation(m, "eq_target")
    eq_target[...] = gp.Sum(t, m_out[t] * dt) / 1000.0 + short >= D_req * n_t * dt / 1000.0

    # Carbon-intensity cap on grid imports per tonne of methanol delivered.
    if par.CI_max is not None:
        eq_ci = gp.Equation(m, "eq_ci")
        eq_ci[...] = gp.Sum(t, eps[t] * P_imp[t] * dt) <= par.CI_max * gp.Sum(t, m_out[t] * dt) / 1000.0

    # Green-power matching within windows of match_window_h hours.
    if par.match_window_h and (par.phi_import is not None or par.phi_spill is not None):
        n_w = int(np.ceil(n_t / par.match_window_h))
        w = gp.Set(m, "w", records=[f"w{j:04d}" for j in range(n_w)], description="Matching windows")
        tw = gp.Set(
            m, "tw", domain=[t, w],
            records=[(ts, f"w{i // par.match_window_h:04d}") for i, ts in enumerate(data.time_steps)],
        )
        if par.phi_import is not None:
            eq_match_imp = gp.Equation(m, "eq_match_imp", domain=w)
            eq_match_imp[w] = gp.Sum(t.where[tw[t, w]], P_imp[t]) <= par.phi_import * gp.Sum(t.where[tw[t, w]], load)
        if par.phi_spill is not None:
            eq_match_spill = gp.Equation(m, "eq_match_spill", domain=w)
            eq_match_spill[w] = gp.Sum(t.where[tw[t, w]], P_exp[t] + P_curt[t]) <= par.phi_spill * gp.Sum(
                t.where[tw[t, w]], cf[t] * C_wind + cf_pv[t] * C_pv
            )

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
        t,
        (
            (price[t] + par.grid_fee + par.tax_CO2 * eps[t]) * P_imp[t]
            - price[t] * P_exp[t]
            + par.c_tiebreak_grid * (P_imp[t] + P_exp[t])
        )
        * dt,
    )
    cost_var = gp.Sum(
        t,
        (
            par.c_water_H2 * m_H2[t]
            + par.c_tiebreak_PEM * P_PEM[t]
            + par.c_CO2_feed * par.gamma_CO2 / 1000.0 * m_M[t]
            - par.p_MeOH / 1000.0 * m_out[t]
            - par.p_H2_sale * h_sell[t]
        )
        * dt,
    )
    cost_start = par.c_cold_start_PEM * C_PEM * gp.Sum(t, y_P[t]) + cost_start_M
    total_cost = cost_grid + cost_var + cost_start + cost_down + par.c_short_MeOH * short

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
        "parameters": {"cf": cf, "cf_pv": cf_pv, "price": price, "eps": eps, "soc0": soc0, "h0": h0},
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
            **extra_vars,
        },
        "short": short,
    }
    return m, model, relaxed, symbols


def solve(
    model,
    solver: str = "cplex",
    gap: float = 0.005,
    time_limit: float = 600.0,
    threads: int = 0,
    output=None,
    abs_gap: Optional[float] = None,
):
    """Solve and return (wall-clock time [s], GAMSPy solve summary)."""

    opts = {"relative_optimality_gap": gap, "time_limit": time_limit, "threads": threads}
    if abs_gap is not None:
        opts["absolute_optimality_gap"] = abs_gap
    t0 = time.perf_counter()
    summary = model.solve(solver=solver, options=gp.Options(**opts), output=output)
    return time.perf_counter() - t0, summary


def hourly_results(data: SchedulingData, design: PlantDesign, par: OperatingParams, symbols) -> pd.DataFrame:
    """Hourly levels of all dispatch variables."""

    df = pd.DataFrame(
        {"t": data.time_steps, "cf": data.cf_wind, "cf_pv": data.cf_pv, "price": data.price, "eps": data.eps_grid}
    )
    for name, var in symbols["variables"].items():
        dense = var.toDense()
        df[name] = np.zeros(len(df)) if dense is None else np.asarray(dense, dtype=float)
    df["P_wind"] = df["cf"] * design.C_wind_MW
    df["P_PV"] = df["cf_pv"] * design.C_PV_MW
    df["P_sb"] = par.f_sb_PEM * design.C_PEM_MW * df["s_P"]
    df["P_MeOH"] = df["P_S"] if "P_S" in df else par.SP_MeOH / 1000.0 * df["m_M"]
    if "m_D" in df:
        df["Q_syn"] = par.q_syn / 1000.0 * df["m_M"]
        df["Q_rls"] = par.q_reb / 1000.0 * (df["m_D"] - df["z_F"])
    return df


def kpis(df: pd.DataFrame, design: PlantDesign, par: OperatingParams) -> dict:
    """Horizon totals from the hourly results."""

    dt = par.dt_h
    hours = len(df) * dt
    meoh_t = df["m_M"].sum() * dt / 1000.0
    out_t = df["m_D"].sum() * dt / 1000.0 if "m_D" in df else meoh_t
    emis_t = (df["eps"] * df["P_imp"]).sum() * dt
    res = {
        "MeOH_t": meoh_t,
        "MeOH_vs_requirement": out_t / (design.demand_kgph * hours / 1000.0),
        "MeOH_load_factor": meoh_t / (design.m_MeOH_nom_kgph * hours / 1000.0),
        "H2_prod_t": df["m_H2"].sum() * dt / 1000.0,
        "H2_sold_t": df["h_sell"].sum() * dt / 1000.0,
        "import_MWh": df["P_imp"].sum() * dt,
        "export_MWh": df["P_exp"].sum() * dt,
        "curtail_MWh": df["P_curt"].sum() * dt,
        "wind_MWh": df["P_wind"].sum() * dt,
        "pv_MWh": df["P_PV"].sum() * dt,
        "emissions_tCO2": emis_t,
        "CI_tCO2_per_tMeOH": emis_t / out_t if out_t > 0 else float("nan"),
        "PEM_on_h": df["u_P"].sum() * dt,
        "PEM_standby_h": df["s_P"].sum() * dt,
        "PEM_cold_starts": df["y_P"].sum(),
        "MeOH_on_h": df["u_M"].sum() * dt if "u_M" in df else hours,
        "MeOH_starts": df["y_M"].sum() if "y_M" in df else 0.0,
        "grid_cost_USD": (df["price"] * (df["P_imp"] - df["P_exp"])).sum() * dt,
    }
    if "m_D" in df:
        dist_MWh = (df["W_F"] + df["P_eb"]).sum() * dt
        rls_MWh = df["Q_rls"].sum() * dt
        res.update({
            "MeOH_delivered_t": out_t,
            "crude_swing_t": (df["I_c"].max() - df["I_c"].min()) / 1000.0,
            "FVC_on_h": df["d_F"].sum() * dt,
            "FVC_MWh": df["W_F"].sum() * dt,
            "eboiler_MWh": df["P_eb"].sum() * dt,
            "dist_kWh_per_kg": 1000.0 * dist_MWh / (1000.0 * out_t) if out_t > 0 else float("nan"),
            "syn_heat_MWh": df["Q_syn"].sum() * dt,
            "FVC_heat_to_TES_MWh": df["Q_str"].sum() * dt,
            "TES_release_MWh": rls_MWh,
            "heat_dumped_MWh": df["Q_dump"].sum() * dt,
            "TES_full_cycles": rls_MWh / design.E_TES_MWh if design.E_TES_MWh > 0 else 0.0,
            "dist_power_cost_USD": (df["price"] * (df["W_F"] + df["P_eb"])).sum() * dt,
        })
    return res

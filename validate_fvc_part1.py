"""
Check the linear FVC + thermal-store block against Qi et al. (2025), scenario b.

Scenario b of Qi et al. (AIChE J 71:e18750) is a stand-alone methanol/water
column (55 kmol/h methanol, reboiler 1167 kW) under Gansu time-of-use prices:
16 off-peak hours per day at 75 USD/MWh, and the FVC off for the 8 peak hours.
Their MINLP (ALAMO surrogates, DICOPT) gives an optimal compressor of 679 kW,
and the FVC beats a two-stage MVR (295 kW, all day) above a flat price of
115.3 USD/MWh.

Here the same day is solved with add_downstream_section(): synthesis output is
fixed at the column throughput, no synthesis heat, no store losses. The
compressor peak is minimised, which is what the 24-h periodic schedule needs.
"""

from __future__ import annotations

import gamspy as gp

from ptm_scheduling_gamspy import OperatingParams, PlantDesign, add_downstream_section

D_KGPH = 55.0 * 32.04  # methanol in the distillate [kg/h]
OFF_PEAK_PRICE = 75.0  # USD/MWh
PEAK_HOURS = range(16, 24)  # FVC off
W_MVR_MW = 0.295  # two-stage MVR, Qi et al. Fig. 4


def main() -> None:
    design = PlantDesign(m_MeOH_nom_kgph=D_KGPH, D_MeOH_kgph=D_KGPH, C_crude_kg=0.0, E_TES_MWh=100.0, W_FVC_MW=5.0)
    par = OperatingParams(distillation="explicit", q_syn=0.0, loss_TES=0.0)

    hours = [f"h{i:02d}" for i in range(24)]
    m = gp.Container()
    t = gp.Set(m, "t", records=hours)
    first = gp.Parameter(m, "first", domain=t, records=[(hours[0], 1.0)])
    last = gp.Parameter(m, "last", domain=t, records=[(hours[-1], 1.0)])
    peak = gp.Parameter(m, "peak", domain=t, records=[(hours[i], 1.0) for i in PEAK_HOURS])
    m_syn = gp.Variable(m, "m_syn", domain=t, type="Positive")
    m_syn.fx[t] = D_KGPH

    v, load, cost_cw = add_downstream_section(m, t, first, last, design, par, m_syn, dt=1.0)
    W_pk = gp.Variable(m, "W_pk", type="Positive")
    eq_off = gp.Equation(m, "eq_off", domain=t)
    eq_off[t] = v["d_F"][t] <= 1 - peak[t]
    eq_pk = gp.Equation(m, "eq_pk", domain=t)
    eq_pk[t] = W_pk >= v["W_F"][t]
    energy_cost = gp.Sum(t, OFF_PEAK_PRICE * load)
    model = gp.Model(
        m, "fvc_check", equations=m.getEquations(), problem="MIP", sense=gp.Sense.MIN,
        objective=W_pk + 1e-6 * energy_cost,
    )
    model.solve(solver="cplex", options=gp.Options(relative_optimality_gap=1e-9))

    w_pk = W_pk.toValue()
    e_day = float(v["W_F"].records["level"].sum())
    p_be = OFF_PEAK_PRICE * e_day / (W_MVR_MW * 24.0)
    cw = float(v["Q_dump"].records["level"].sum())
    print(f"compressor peak  {1000 * w_pk:7.1f} kW   (Qi et al. 2025: 679 kW)")
    print(f"FVC energy/day   {e_day:7.3f} MWh, FVC on {int(v['d_F'].records['level'].sum())} h, heat dumped {cw:.3f} MWh")
    print(f"MVR break-even   {p_be:7.1f} USD/MWh (Qi et al. 2025: 115.3), electricity only")


if __name__ == "__main__":
    main()

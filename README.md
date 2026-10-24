# Flexible power-to-methanol scheduling MILP (GAMSPy)

Hourly mixed-integer linear scheduling model for a wind-powered power-to-methanol (PtM) plant with bidirectional grid exchange, a battery (BESS) and a compressed-hydrogen buffer, written in [GAMSPy](https://gamspy.readthedocs.io).

The model is the operational layer of a design-under-uncertainty study. Installed capacities are inputs, and the model returns the cost-optimal hourly dispatch with full knowledge of the horizon (perfect foresight).

## Model scope

| Unit | Representation |
|---|---|
| Wind farm | Capacity factor × installed capacity, with curtailment |
| Grid | Import and export up to the connection limit; import pays price + fee + carbon tax |
| BESS | Energy balance with charge/discharge efficiency, self-discharge, SOC window and C-rate |
| PEM electrolyser | On / warm-standby / off states, minimum load, standby power and cold-start cost. The part-load H₂ curve is concave and is represented exactly by its piecewise-linear upper envelope, without SOS2 binaries |
| H₂ tank | Inventory balance with compression power; optional H₂ sales |
| Methanol synthesis | On/off, minimum load, ramp limits, minimum up/down time, start-up cost and a soft production target |
| Carbon rule | Grid-import emissions per tonne of methanol ≤ `CI_max`, with an hourly grid emission factor |
| Horizon end | End-of-horizon BESS and H₂ levels ≥ initial levels (optional) |

### Differences from the MasCOR operational model

The examples reuse the scenarios and designs of MasCOR (Kim et al., 2026). The operational model of that work keeps methanol synthesis at constant nameplate load and has no electrolyser states. This model adds the following:

- Flexible methanol synthesis with unit-commitment constraints.
- Electrolyser on/standby/off states with a part-load efficiency curve.
- Explicit wind curtailment. Because curtailment is a decision variable, simultaneous import and export is never optimal, so the grid-direction binaries are dropped.
- Hourly emission factors and a carbon-intensity cap. `CI_max = 1.435802` tCO₂/tMeOH reproduces MasCOR's net-negative rule.
- Negative electricity prices are kept rather than clipped at zero.

## Formulation (main constraints)

Power balance:

$$CF_t C^{W} - P^{curt}_t + P^{imp}_t + P^{dis}_t = P^{exp}_t + P^{ch}_t + P^{PEM}_t + f^{sb} C^{PEM} s_t + P^{comp}_t + e^{MeOH}\dot m_t$$

Electrolyser:

$$u_t + s_t \le 1,\qquad f^{min} C^{PEM} u_t \le P^{PEM}_t \le C^{PEM} u_t,\qquad \dot m^{H_2}_t \le a_k P^{PEM}_t + b_k C^{PEM} u_t \;\; \forall k$$

Methanol synthesis:

$$f^{M}\dot m^{nom} v_t \le \dot m_t \le \dot m^{nom} v_t,\qquad v_t - v_{t-1} = y_t - z_t,\qquad \sum_{\tau=t-UT+1}^{t} y_\tau \le v_t,\qquad \sum_{\tau=t-DT+1}^{t} z_\tau \le 1 - v_t$$

Carbon-intensity cap:

$$\sum_t \varepsilon_t P^{imp}_t \Delta t \le CI^{max} \sum_t \dot m_t \Delta t$$

The objective minimises grid cost, carbon tax, variable costs, start-up costs and the production shortfall penalty over the horizon. Unit conversions are omitted above and are written out in `ptm_scheduling_gamspy.py`.

## Installation

```bash
pip install -r requirements.txt
```

The example instances (17k–263k equations) exceed GAMSPy's size limits without a license. Academic licenses are free from GAMS. The default solver is CPLEX; pass `solver=` to `solve()` to use another MIP solver.

## Quick start (synthetic data)

```python
import numpy as np
from ptm_scheduling_gamspy import (
    OperatingParams, PlantDesign, SchedulingData,
    build_model, hourly_results, kpis, solve,
)

rng = np.random.default_rng(0)
n = 168
data = SchedulingData.from_arrays(cf_wind=rng.uniform(0, 1, n), price=rng.uniform(0, 100, n), eps_grid=0.5)
design, par = PlantDesign(), OperatingParams()

m, model, relaxed, symbols = build_model(data, design, par)
wall_s, _ = solve(model)
print(model.objective_value, kpis(hourly_results(data, design, par, symbols), design, par))
```

## Examples on the MasCOR dataset

`run_ptm_examples.py` solves three cases:

1. A 576-h synthetic scenario for Skive (Denmark) using the MasCOR-sampled design, with flexible synthesis.
2. The same scenario with rigid synthesis (off or nameplate).
3. A full historical year, 2023 at Skive (8760 h), with a demonstration design.

To run them:

1. Download the `dataset/` folder (about 4.5 GB) of the [MasCOR Kaggle dataset](https://www.kaggle.com/datasets/jeongdongkim/dataset-for-mascor-framework).
2. Point `MASCOR_DATASET` at it:

   ```bash
   export MASCOR_DATASET=/path/to/dataset   # PowerShell: $env:MASCOR_DATASET="C:\path\to\dataset"
   python run_ptm_examples.py
   ```

Hourly results and a summary are written to `results/`.

Reference solve times (CPLEX, default threads, 0.5 % relative gap):

| Case | Horizon | Equations | MILP | LP relaxation |
|---|---|---|---|---|
| Skive scenario 0, flexible | 576 h | 17,285 | 6 s | 0.3 s |
| Skive scenario 0, rigid | 576 h | 17,285 | 4 s | 0.4 s |
| Skive 2023 | 8760 h | 262,805 | 207 s | 4 s |

## Status and limitations

- **Placeholder parameters.** Values marked `ASSUMPTION` in `OperatingParams` still need plant- or literature-specific calibration. This applies to start-up costs, minimum up/down times and the intermediate points of the PEM curve.
- **Constant emission factor.** The examples use 0.5 tCO₂/MWh, as in MasCOR. The carbon cap binds in all example cases, so results depend strongly on this value.
- **Single-horizon model.** Minimum up/down history and the cumulative constraints (carbon cap, production target) apply within one horizon only, so the model is not yet set up for rolling-horizon use.
- **BESS at negative prices.** At negative prices the battery may charge and discharge in the same hour to dissipate energy. Set `bess_exclusive=True` to prevent this.

## Units

Time is in h; power in MW; electrical energy in MWh; H₂ and methanol flows in kg/h; H₂ inventory in kg; prices in USD/MWh; emission factors in tCO₂/MWh.

## Reference

Kim, J. et al. (2026). MasCOR: machine-learning-assisted co-optimization of e-fuel system design and operation. arXiv:2603.03484. Code: https://github.com/PSEKJD/mascor

## License

MIT

# P5-D0.2 single-sample unconditional coordinate-flow overfit

Single-sample overfit: `True`. Attribution: `passed`.

- Final velocity MSE: `1.84458768e-11` (required `<= 1.0e-05`)
- Final endpoint periodic RMS: `1.29734269e-06` (required `<= 0.005`)
- `loss_curve.csv` records every update's loss, endpoint RMS, coordinate-head gradient norm, output norm, and target-velocity norm.
- `coordinate_components.csv` records every final node/dimension prediction and target.

No condition input, model-capacity change, extra training, harmonic module, oracle, or subsequent Gate is used. P5-D1 remains prohibited.

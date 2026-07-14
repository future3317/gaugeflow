# TensorOrbit-JARVIS-v2 external oracle qualification preparation

## Status

`prepared_commit_required_before_external_training`. The v2 candidate split remains inactive for GaugeFlow.
This preparation creates matched **external oracle** input manifests only; it
does not start GMTNet, the SE(3)-Transformer, PiezoJet, GaugeFlow, S2, or a
4,000/499/499 run.

## Frozen data identity

- Candidate split SHA-256: `db5223dd7a57097648c956c710a7794bc1f09228f12a4ab4ce1b64b1a21c24fd`
- Protocol SHA-256: `b1448671a7333e1cd2dbb5451d47ebdd794a5a00b173ea4f31243f8f7bda7244`
- Split counts: `{'train': 4000, 'val': 499, 'test': 499}`
- Oracle manifests: `{'gmtnet': '7ecd7cc61d17798f52244ef0c186f1ef9747acaa43915869377314b69ef9508e', 'se3_transformer_rank3': '0e2dab83fd7bf0263b9df7f379871300e7ef9c9033c86a001489ca434ba56848'}`

## Activation boundary

Before either external training job starts, pin its source repository and
commit, environment lock, entrypoint, this protocol/manifest commit, and the
same v2 split hash. Both GMTNet and the architecture-distinct e3nn
SE(3)-Transformer must complete matched v2 validation before any frozen oracle
ensemble is qualified. PiezoJet is explicitly not the primary oracle.

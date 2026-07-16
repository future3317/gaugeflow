# Frozen S1a-I0 v1/v1.1/v1.2 runner mismatch

The archived closure runner used `learning_rate=1e-3`, `weight_decay=0`, and
`ema_decay=0.95` directly in code while the corresponding JSON protocols listed
`2e-4`, `1e-6`, and `0.999`. The observed failures remain valid implementation
failures, but they are not evidence for the optimizer values printed in those
JSON files. The v1.3 successor removes the hard-coded values and reads every
training parameter from its versioned protocol. Historical result JSON files
and thresholds are not modified.

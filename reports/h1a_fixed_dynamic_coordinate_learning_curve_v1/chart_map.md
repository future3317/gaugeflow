# Chart map

| Artifact | Analytical question | Family / form | Fields | Supported claim | Palette policy |
|---|---|---|---|---|---|
| `figures/h1a_all_pair_clean_topology_v2` | Is noisy topology disrupted, predictive, and causally useful? | Three-panel ordered line/dot comparison | noise time, disagreement, residual improvement, AUC, explained fraction | Clean topology is informative and predictable, but the frozen plug-in correction fails | hard two-root cap: blue/orange plus grey controls; markers and line styles duplicate series identity |
| `figures/h1a_fixed_dynamic_learning_curve_v1` | Does the unchanged model plateau after one data pass? | Exposure dot-line, log endpoint curve, smoothed training trace, log-gradient trace | passes, validation ratio, endpoint RMS, batch loss, module gradient norm | A second pass improves validation by 9.6876%, leaving the preregistered classification ambiguous | hard two-root cap: blue/orange plus grey references; markers and line styles duplicate series identity |

Both figures are exported as PNG for Markdown/quick inspection and vector PDF for the paper. The PNGs were inspected at original resolution after export; titles, legends, labels, axes, and annotations do not overlap or clip.

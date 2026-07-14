# A4.6 upstream FlowMM reference audit

Status: read-only reference inspection; GaugeFlow does not import FlowMM at runtime.

The preserved FlowMM baseline identifies upstream main commit `6a96aec3b6eba89f6fa07436f0c8837979abb285` and local head `201c5a6f095739bbed61bd0f5f6c381dd41a5f85`.

Inspected read-only checkout: `outputs\gate_a4_generator_substrate_v1\upstream_flowmm_readonly` (62 Python files).
Static type-path tokens found: 3; decoder/sampler tokens found: 3.
FlowMM declares a simplex atom-type manifold: True; its inverse one-hot decoder is argmax-plus-one: True.
Source-equivalent decoder microtest for B/N/In one-hot states produced atomic numbers [5, 7, 49] (expected [5, 7, 49]).
A full upstream two-endpoint flow/sampler run is blocked: importing the pinned checkout requires the DiffCSP `torch_scatter` extension, which is not installed in this environment. The decoder microtest is not reported as a full numerical baseline result.

Decision use: if a later pinned, runnable upstream reference passes the same endpoint-ID type test while standalone GaugeFlow does not, port only the verified manifold/path/decoder definition under a new protocol. Do not restore FlowMM as a runtime dependency.

# Full-Alex assignment pretraining interface v1

Status: **FAIL (frozen)**. No model training started.

The preregistered shell-based CVP certificate rejected 918 of the 540,164
training graphs, covering 28,632 directed pairs. An independent float64 sphere
decoder showed that every true closest image was nevertheless inside the
initial 27-image cube (maximum exact shell 1). The failure is therefore an
overconservative certificate, not corrupt Alex-MP data: the bound

\[
d_{\rm best}\le \sigma_{\min}(L)(m+\tfrac12)
\]

throws away directional information and becomes weak for valid elongated
cells. The failed protocol, implementation commit, exception and aggregate
diagnostic are preserved in `result.json`. Its shell threshold was not changed
and none of the affected structures was deleted.

The separately frozen successor replaces the isotropic shell expansion with a
dual-lattice coordinate bound that enumerates exactly the finite integer box
containing every possible minimizer. This is a new interface protocol, not a
reinterpretation of v1.

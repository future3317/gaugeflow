# S0.1 audit-definition amendment

The immutable first attempt at
`reports/paper_s0_mathematical_qualification_v1/` failed two source scans while
all tests and static checks passed. The scan concatenated its own
`s0_audit.py`; consequently the forbidden-pattern literals inside the audit
code matched themselves.

S0.1 changes only the scan domain: production runtime modules are scanned and
the audit runner itself is excluded. No production model code, test, threshold,
paper hash, or successor rule changes. The first failed attempt remains
tracked and must not be overwritten.

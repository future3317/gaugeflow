# GaugeFlow-base capacity execution smoke v1

All three preregistered capacities passed the CUDA execution qualification on
an RTX 4090 with the same effective batch size of 64. This is an engineering
qualification only; it does not claim that a larger network learns a better
generative law.

| candidate | parameters | physical / accumulation | graphs/s | peak MiB | result |
|---|---:|---:|---:|---:|---|
| small | 34,284,207 | 64 / 1 | 238.73 | 14,907.40 | qualified |
| base | 57,682,095 | 32 / 2 | 138.01 | 12,738.24 | qualified |
| large | 97,580,719 | 16 / 4 | 70.60 | 10,470.10 | qualified |

The lower measured allocation of the larger candidates is caused by the
smaller physical microbatch, not by a smaller model. Graph-weighted gradient
accumulation preserves an effective batch of exactly 64 and performs one
optimizer, clipping and EMA update per effective batch. All losses and
gradients were finite and all candidates remained below the frozen 22 GiB
allocation ceiling.

The next Gate changes capacity only and gives every candidate exactly 540,164
graph presentations. It selects the smallest candidate within frozen quality
margins of the best eligible candidate, so a 98M model is retained only if its
teacher-forced and free-running gain is material.

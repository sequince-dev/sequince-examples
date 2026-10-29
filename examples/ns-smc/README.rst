NS-SMC
======

Nested sampling via sequential Monte Carlo (NS-SMC) estimates evidence and
posterior expectations by sampling through successive likelihood shells.
The method is described by `Salomone et al. (2025)
<https://doi.org/10.1093/jrsssb/qkaf015>`_ in *Unbiased and consistent nested
sampling via sequential Monte Carlo*. The paper distinguishes fixed-threshold
NS-SMC from adaptive NS-SMC (ANS-SMC). Sequince's helper uses adaptive
thresholds by default; these do not inherit fixed-schedule unbiasedness
guarantees.

Start with the Gaussian example for basic usage. The spike-and-slab example
compares nested and tempered SMC on a model with a phase transition.
Both examples run from this directory (``ns-smc``).

.. toctree::
   :maxdepth: 1

   gaussian
   spike_and_slab

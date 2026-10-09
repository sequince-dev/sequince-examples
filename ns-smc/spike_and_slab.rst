Spike-and-slab example
======================

`run_spike_and_slab.py <run_spike_and_slab.py>`_ compares tempered SMC
and NS-SMC on a spike-and-slab Gaussian mixture with a uniform prior on the
ten-dimensional unit ball, following §6.1 of
`Salomone et al. <https://doi.org/10.1093/jrsssb/qkaf015>`_. The narrow central
spike occupies little prior volume but dominates the posterior. As the
temperature increases along the tempered path, probability shifts sharply
from the broad slab to the spike: this is the phase transition. NS-SMC uses
the known global likelihood maximum to bound the remaining evidence when
stopping.

The comparison uses MiniPCN for standard, persistent, and nested SMC, with
exact constrained-prior draws as a diagnostic control.

Run from this directory (``ns-smc``), using an environment with Sequince,
NumPy, MiniPCN, SciPy, and Matplotlib installed:

.. code-block:: bash

   python run_spike_and_slab.py

Outputs in ``outputs/spike_and_slab/`` include sampling-path and posterior
plots, a JSON summary of evidence and likelihood costs, and NPZ sample data.
Use ``--plot-paths-only`` to generate the idealized reference paths without
running samplers, or ``--seeds`` and ``--particles`` to change the run size.
Use ``--outdir`` to keep separate experiments.

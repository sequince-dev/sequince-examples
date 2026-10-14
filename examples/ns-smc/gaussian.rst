Gaussian example
================

`example.py <example.py>`_ estimates evidence and posterior means for a
two-dimensional Gaussian likelihood centred on ``(1, -1)``, with a Gaussian
prior of standard deviation 5. It prints the estimates alongside their
analytic values.

Run from this directory (``ns-smc``), using an environment with Sequince,
NumPy, and MiniPCN installed:

.. code-block:: bash

   python example.py

The script defines normalized log-prior and log-likelihood functions:

.. literalinclude:: example.py
   :language: python
   :start-at: def log_prior(
   :end-before: def main(

It then draws initial particles from the prior, constructs and runs the
sampler, and compares the evidence and posterior mean with analytic values:

.. literalinclude:: example.py
   :language: python
   :pyobject: main

Each level discards a likelihood shell, adds its evidence contribution,
and resamples and mutates the survivors within the new constraint. Here,
the threshold is the weighted median likelihood. Stopping integrates the
remaining live tail when its estimated evidence is sufficiently small.

Use ``result.log_evidence`` for evidence and the returned posterior weights
in averages or histograms. The posterior combines all stored shells;
``raw.final_state`` is a constrained prior, not the posterior. The iteration
limit can leave a run incomplete, which is why the example checks
``result.complete`` before using it.

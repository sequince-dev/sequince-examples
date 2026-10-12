Waste-free sequential Monte Carlo
================================

`run_waste_free_smc.py <run_waste_free_smc.py>`_ demonstrates waste-free
sequential Monte Carlo (SMC), as described by `Dau & Chopin`_, alongside a
standard resample-move SMC run. Both sample a four-dimensional correlated
Gaussian likelihood with a uniform prior on ``[-10, 10]^4``.

Run from this directory using an environment with Sequince, NumPy and
Matplotlib installed:

.. code-block:: bash

   python run_waste_free_smc.py

Wrap a mutation step to retain its full chains:

.. code-block:: python

   WasteFreeResampleMove(
       MetropolisHastingsMutation(posterior, GaussianRandomWalk(scale=0.7)),
       chain_length=100,
   )

With N = 10,000 input particles and chain length P = 100, the waste-free
step resamples M = N / P = 100 starts, runs P - 1 = 99 transitions per
chain and retains all 100 samples, including each start and repeats after
rejection. The output has 10,000 uniformly weighted particles.
``chain_length`` includes the start and must divide N.

For comparison, standard SMC explicitly resamples and mutates:

.. code-block:: python

   MultinomialResampling(always_resample=True)
   MetropolisHastingsMutation(
       posterior, GaussianRandomWalk(scale=0.7), n_steps=99
   )

The standard run maintains 100 particles and keeps each chain's endpoint.
Both runs use the same target, proposal, tempering ESS ratio and 100 chains
with 99 transitions each per stage. This matches mutation work per stage;
initial population sizes and adaptive temperature schedules differ, so
stage counts and total target evaluations can differ. The script logs
likelihood evaluation counts alongside posterior moments. Retained
waste-free samples are correlated, and their count is not an
independent-sample effective sample size. One run of each method illustrates
usage, rather than establishing relative statistical efficiency.

``LoggingStep`` reports temperature and log evidence at each stage. The
final log summary compares posterior counts, stage counts, likelihood
evaluations and evidence estimates, and quotes the retained-count ratio.
The single figure, ``waste_free_history.png``, compares the inverse-temperature
schedules. Stage numbers match the log messages. The legend and logs quote
the final posterior counts: 100 endpoints for standard SMC and 10,000
retained samples for waste-free SMC. Both populations have uniform weights;
the larger retained count does not imply 100 times as many independent
samples.

The target has zero mean and covariance
``covariance[i, j] = 0.8 ** abs(i - j)``. Posterior means should approach
zero and marginal variances should approach one. The evidence reference is
``-4 * log(20)`` (approximately -11.983), neglecting the tiny Gaussian mass
outside the prior box. Both densities include their normalization constants.

The target densities and complete sampler setup are shown below:

.. literalinclude:: run_waste_free_smc.py
   :language: python
   :linenos:

.. _Dau & Chopin: https://arxiv.org/abs/2011.02328

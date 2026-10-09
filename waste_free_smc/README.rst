Waste-free sequential Monte Carlo
================================

`run_waste_free_smc.py <run_waste_free_smc.py>`_ samples a four-dimensional
standard Gaussian likelihood with a uniform prior on ``[-10, 10]^4`` using
waste-free sequential Monte Carlo (SMC), as described by `Dau & Chopin`_.
``LoggingStep`` reports the temperature and log evidence at each stage.
The script also logs the final temperature, log evidence and posterior
moments, and saves ``waste_free_history.png`` with three panels: the
temperature schedule, log evidence with its reference and retained chains.
Stage numbers in the plots match the log messages.

The chain panel shows the first coordinate of three final-stage chains.
Squares mark resampled starts, crosses mark endpoints and every dot is a
retained state. Flat segments show repeated states after rejected proposals.
Waste-free SMC keeps all 100 positions per chain, rather than just the
endpoints. The full population contains 100 such chains, giving 10,000
retained particles; only three trajectories are shown for clarity.

Run from this directory using an environment with Sequince, NumPy and
Matplotlib installed:

.. code-block:: bash

   python run_waste_free_smc.py

The script defines normalized log-likelihood and log-prior functions:

.. literalinclude:: run_waste_free_smc.py
   :language: python
   :start-at: def log_likelihood(
   :end-before: def main(

It then draws particles from the prior and combines adaptive tempering with
``WasteFreeResampleMove``, wrapping a Metropolis-Hastings mutation with a
fixed Gaussian random-walk proposal:

.. literalinclude:: run_waste_free_smc.py
   :language: python
   :pyobject: main

At each stage, adaptive tempering reweights all N = 10,000 particles and
updates the evidence estimate. The waste-free step resamples M = 100 chain
starts, runs P - 1 = 99 transitions per chain and retains all P = 100
states, including each start and repeated states after rejection. The
output contains N = M * P uniformly weighted particles. ``chain_length``
includes the start and must divide N.

Retained states within each chain are correlated: the particle count is
not an independent-sample effective sample size. There is no additional
resampling inside the waste-free step after the chains run.

The final temperature should reach one, posterior means should approach
zero and marginal variances should approach one. The reference log
evidence is ``-4 * log(20)`` (approximately -11.983), neglecting the tiny
Gaussian mass outside the prior box. Both densities include their
normalization constants so the evidence has the intended scale.

.. _Dau & Chopin: https://arxiv.org/abs/2011.02328

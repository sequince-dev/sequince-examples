Adaptive mutation steps
=======================

In sequential Monte Carlo (SMC), resampling can produce many copies of the
same particle. Mutation moves help these copies explore the current target
distribution, but choosing how many moves to run involves a tradeoff: too few
can leave the population poorly explored, while too many spend computation
on moves that may add little useful exploration.

A fixed number of mutation steps may also be a poor fit throughout a run.
As tempering moves from the prior towards the posterior, the target becomes
more concentrated and the acceptance rate can change. This example adapts
the number of attempted moves using the previous stage's acceptance rate,
aiming for a chosen number of accepted moves per particle. Lower acceptance
requests more attempts at the next stage, up to a configurable limit. This
provides a simple way to adjust mutation effort without choosing a separate
step count for every stage. Accepted moves are only a rough guide to
exploration, so the example also plots the adaptation and posterior for
inspection.

``adaptive_mutation_steps.py`` samples the same Rosenbrock likelihood
family as the basic examples, using eight dimensions by default, a normalized
uniform prior on ``[-5, 5]^8`` and an unnormalized likelihood:

.. math::

   \log L(\boldsymbol{x}) = -\frac{1}{s}\sum_{j=1}^{d-1}
   \left[100(x_{j+1} - x_j^2)^2 + (1 - x_j)^2\right].

The default scale is ``s = rosenbrock_scale = 1``. Evidence refers to this
likelihood convention; the example does not provide an analytic reference.
The likelihood reads the scale and the prior reads the bounds from
``context.shared``.

Setup and running
-----------------

Use Python 3.11 or later with Sequince's plotting and HDF5 dependencies::

   python -m pip install 'sequince[corner,hdf5]'

Run from the ``adaptive_mutation_steps`` directory::

   python adaptive_mutation_steps.py

Edit the parameters near the top of the script to experiment. Outputs go to
``outdir`` within this example's directory.
Running again replaces those outputs. The example uses a seeded NumPy
generator, like the basic examples; its random stream may differ from older
versions that initialized an ``orng.RandomGenerator`` directly.

How adaptation works
--------------------

The sampler draws equal-weight particles from the prior, then repeats
adaptive tempering, systematic resampling and differential evolution mutation.
The tempering ESS target is ``0.7 * n_particles``; resampling runs at every
stage. Temperature adaptation and mutation-budget adaptation are separate.

Wrap the mutation step with an acceptance-based policy:

.. code-block:: python

   adaptive_mutation = AdaptMutationSteps(
       DifferentialEvolutionMutation(posterior),
       policy=AcceptanceRatePolicy(
           target_steps=target_steps,
           min_steps=min_steps,
           max_steps=max_steps,
       ),
       initial_steps=initial_steps,
       min_steps=min_steps,
       max_steps=max_steps,
   )

``target_steps=5`` targets five accepted moves per particle, rather than five
attempts. Each stage executes a fixed budget, initially ``5 * n_dimensions``
(40 by default). After the complete block, the policy measures acceptance
``a_t = n_accepted / n_attempted`` and selects the next budget:

.. math::

   m_{t+1} = \operatorname{clip}\left(
       \left\lceil\frac{5}{\max(a_t, 10^{-6})}\right\rceil,
       1, 128\right).

Thus 50% acceptance requests ten steps next time, and 10% requests fifty.
The budget is frozen before the next mutation block starts; adaptation never
stops a block based on its own trajectory. Logs show both the completed
budget and the next decision. The final decision is unused once the sampler
reaches ``beta=1``.

Inspect the plots and results
-----------------------------

The script saves:

* ``adaptive_mutation_history.png``: three panels showing executed mutation
  steps, observed acceptance probability, and accepted moves per particle
  (steps times acceptance). The dashed line marks the target of five accepted
  moves. Iteration numbers start at zero and match the logs. The
  ``AdaptiveMutationPlot`` callback updates this figure during the run.
* ``history.png``: temperature and log evidence versus SMC iteration.
* ``corner.png``: marginal and joint posterior distributions.
* ``posterior.npz``: final particles, normalized weights and log evidence.
* ``result.h5``: the saved SMC result, including completed mutation metrics.

Read the first two adaptation panels together: lower acceptance increases
the following stage's budget. Accepted moves need not equal the target,
because acceptance changes between stages, budgets are integers and the
maximum can prevent enough moves. Acceptance alone does not measure mixing
or independent-sample effective sample size. Inspect the posterior and compare
several seeds and particle counts before judging sampling quality.

Try lowering ``max_steps`` to see the budget saturate, or increasing
``target_steps`` to request more mutation work. Keep
``min_steps <= initial_steps <= max_steps`` and ``n_dimensions >= 2``.
Retain weights when loading ``posterior.npz`` for further analysis.

Example results
~~~~~~~~~~~~~~~

:download:`Download script <adaptive_mutation_steps.py>`.

.. figure:: /_generated/results/adaptive-mutation/adaptive_mutation_history.png
   :alt: Executed mutation budgets, acceptance rates, and expected accepted moves across SMC stages.

   Executed mutation budgets, acceptance rates, and expected accepted moves across SMC stages.

.. figure:: /_generated/results/adaptive-mutation/corner.png
   :alt: Weighted marginal and joint distributions for the eight-dimensional Rosenbrock posterior.

   Weighted marginal and joint distributions for the eight-dimensional Rosenbrock posterior.

.. figure:: /_generated/results/adaptive-mutation/history.png
   :alt: Temperature and log evidence through the adaptive mutation run.

   Temperature and log evidence through the adaptive mutation run.

:download:`Run provenance </_generated/results/adaptive-mutation/metadata.json>`.

The full sampler setup is shown below:

.. literalinclude:: adaptive_mutation_steps.py
   :language: python
   :linenos:

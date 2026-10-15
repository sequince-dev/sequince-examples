Rosenbrock posterior
====================

Start with :doc:`gaussian` for the prior, particle initialization, and standard
SMC workflow. This example keeps the same uniform prior and uses a curved
likelihood with differential-evolution mutation and conditional resampling.

Run from this directory with the same dependencies as the Gaussian example::

   python rosenbrock.py

Outputs are written to ``outdir/rosenbrock``.

Define the likelihood
---------------------


``rosenbrock.py`` follows the same workflow with a Rosenbrock log likelihood:

.. math::

    \log L(\boldsymbol{\theta}) = -\sum_{j=1}^{d-1}
    \left[b\left(\theta_{j+1} - \theta_j^2\right)^2
    + \left(a - \theta_j\right)^2\right].

The likelihood reads :math:`a` and :math:`b` from ``context.shared``. With the
default :math:`a=1` and :math:`b=100`, the two-dimensional likelihood concentrates
around a narrow curved valley near :math:`\theta_2 = \theta_1^2`, with a maximum
at :math:`(\theta_1, \theta_2) = (1, 1)`. Inspect the corner plot for this shape
and the history plot for how tempering progresses. This likelihood is
unnormalized: the reported evidence uses exactly the convention above, and the
script provides no analytic reference value.

Configure mutation and resampling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Rosenbrock example also demonstrates more options in the prebuilt helper:

.. code-block:: python

    smc, posterior, metadata = make_standard_bayesian_smc(
        log_prior=log_prior,
        log_likelihood=log_likelihood,
        weighting_criterion="cess",
        target_ess_ratio=0.8,
        resampling="stratified",
        always_resample=False,
        resampling_ess_threshold=0.5,
        mutation="differential-evolution",
        mutation_n_steps=32,
        mutation_kwargs={"gamma_one_probability": 0.1},
    )

``mutation="differential-evolution"`` proposes moves using scaled differences
between other particles in the ensemble. ``mutation_kwargs`` passes options to
the mutation implementation: here ``gamma_one_probability=0.1`` gives each
proposal a 10% chance of using a unit scale instead of the usual
dimension-dependent scale. Each mutation stage takes 32 steps.

``resampling="stratified"`` changes the resampling scheme from the Gaussian
example's default systematic scheme. With ``always_resample=False``, resampling
occurs when the population ESS falls below :math:`0.5N`. By default,
``resample_at_end=True`` also resamples at the final posterior, even if that
threshold has not been crossed. When omitted, ``always_mutate`` follows the
resolved ``always_resample`` value, so mutation runs only on iterations that
resample here. Set ``always_mutate=True`` to run mutation every iteration.

Use ``weighting_criterion="cess"`` with conditional resampling. Conditional ESS
(CESS) measures the effect of a proposed temperature increment given the
current weighted population; ``target_ess_ratio=0.8`` controls that increment.
The separate ``resampling_ess_threshold=0.5`` controls when to resample using the
population ESS. Sequince's ordinary ``"ess"`` tempering criterion requires
resampling every iteration, so simply setting ``always_resample=False`` with
that criterion is invalid.

See the main API reference for ``make_standard_bayesian_smc`` and
``DifferentialEvolutionMutation`` for the available options.

Try changing ``n_particles``, ``mutation_n_steps``, and the seed to assess the
stability of the posterior summaries. Increasing ``target_ess_ratio`` usually
produces smaller tempering increments and more iterations. Increasing particle
count or mutation steps increases computational cost. Compare several runs
before judging whether a setting explores the posterior adequately.

Rosenbrock results
~~~~~~~~~~~~~~~~~~

:download:`Download script <rosenbrock.py>`.

.. figure:: /_generated/results/rosenbrock/corner.png
   :alt: Weighted posterior samples along the curved Rosenbrock valley.

   Weighted posterior samples along the curved Rosenbrock valley.

.. figure:: /_generated/results/rosenbrock/history.png
   :alt: Temperature and log evidence for the unnormalized Rosenbrock likelihood.

   Temperature and log evidence for the unnormalized Rosenbrock likelihood.

:download:`Run provenance </_generated/results/rosenbrock/metadata.json>`.

.. literalinclude:: rosenbrock.py
   :language: python
   :linenos:

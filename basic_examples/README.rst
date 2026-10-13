Basic examples
==============

These examples walk through Bayesian inference with Sequince's standard
Sequential Monte Carlo (SMC) sampler. Start with ``gaussian.py`` to learn the
workflow, then run ``rosenbrock.py`` to explore a curved posterior and
customize mutation and resampling. Each script
contains its own prior, likelihood, and sampler setup. Edit the parameters near
the top of the script to experiment.

Setup and running
-----------------

Use Python 3.11 or later and install Sequince with the optional plotting and
HDF5 dependencies::

    python -m pip install 'sequince[corner,hdf5]'

For installation options, including using a development checkout, see
"Getting started" in the main Sequince documentation.

Run the scripts from this directory::

    python gaussian.py
    python rosenbrock.py

The scripts write to ``outdir/gaussian`` and ``outdir/rosenbrock`` within this
directory.
Running a script again replaces its previous outputs.

1. Define the target distribution
---------------------------------

Sequince needs functions that evaluate the log prior and log likelihood. Both
receive a batch of particles with shape ``(n_particles, n_dimensions)`` and a
``StepContext``, and return one value per particle, with shape
``(n_particles,)``. Sequince passes the context when evaluating the target.
Use ``context.shared`` to read static model parameters supplied to ``smc.run``
through ``shared_context``. For example, both scripts define the prior as:

.. code-block:: python

    def log_prior(particles, context):
        lower_bound = context.shared["lower_bound"]
        upper_bound = context.shared["upper_bound"]
        inside = np.all(
            (particles >= lower_bound) & (particles <= upper_bound), axis=-1
        )
        log_density = -particles.shape[1] * np.log(upper_bound - lower_bound)
        return np.where(inside, log_density, -np.inf)

The Gaussian likelihood reads ``mean`` and ``sigma`` from the same shared
mapping; the Rosenbrock likelihood reads its coefficients ``a`` and ``b``.
This makes the target functions independent of module-level model settings.

For a parameter vector :math:`\boldsymbol{\theta}` with :math:`d` dimensions,
the examples use independent uniform priors between -5 and 5:

.. math::

    \log p(\boldsymbol{\theta}) =
    \begin{cases}
        -d \log 10, & \boldsymbol{\theta} \in [-5, 5]^d, \\
        -\infty, & \text{otherwise}.
    \end{cases}

Including the prior normalization is necessary for a correct evidence estimate.

In ``gaussian.py``, the likelihood is a standard multivariate Gaussian,
including its normalization constant (the defaults are ``mean=0`` and
``sigma=1``). The resulting posterior is a standard
Gaussian restricted to the prior box. Its mean is zero, and its covariance is
very close to the identity for these wide bounds. The script also computes the
analytic evidence, including the Gaussian probability mass inside the box.

2. Initialize particles from the prior
--------------------------------------

Create a seeded NumPy generator, draw particles from the prior, and put them in
a ``ParticleState``:

.. code-block:: python

    rng = np.random.default_rng(seed)
    initial_state = ParticleState(
        particles=rng.uniform(
            lower_bound, upper_bound, size=(n_particles, n_dimensions)
        ),
    )

When ``log_weights`` is omitted, ``ParticleState`` supplies equal normalized
weights, each with log weight :math:`\log w_i = -\log N`, where :math:`N` is
the number of particles (``n_particles``). This is appropriate here
because the particles are drawn from the prior, the initial target distribution.
There is no need to set the weights explicitly. Supply ``log_weights`` when
constructing a state that already has nonuniform importance weights.

Keep using the same generator for the sampler. Changing the initialization to a
different distribution requires accounting for that proposal; see the main
API reference for ``make_standard_bayesian_smc`` and its ``log_proposal`` option.

3. Configure the sampler
------------------------

Use the prebuilt helper to assemble the standard Bayesian SMC workflow:

.. code-block:: python

    smc, posterior, metadata = make_standard_bayesian_smc(
        log_prior=log_prior,
        log_likelihood=log_likelihood,
        target_ess_ratio=0.8,
        mutation="minipcn",
        mutation_n_steps=16,
    )

The sampler moves through tempered distributions

.. math::

    \pi_\beta(\boldsymbol{\theta}) \propto
    p(\boldsymbol{\theta})\,L(\boldsymbol{\theta})^\beta,
    \qquad 0 \leq \beta \leq 1,

where :math:`p` is the prior and :math:`L` is the likelihood. It starts at the
prior with :math:`\beta=0` and finishes at the posterior with :math:`\beta=1`.
Adaptive tempering chooses increments using an effective sample size (ESS)
criterion. Resampling concentrates particles in
regions of higher weight, and mutation moves help the population explore each
intermediate distribution.

The helper returns the SMC runner, a tempered posterior object, and initial
metadata. These scripts use the runner and metadata directly; the posterior
object is also available for further sampling with ``draw_additional_samples``.
See "Configuring the runner" in the main User guide and the helper's API
reference for resampling thresholds, mutation choices, and other options.

4. Run SMC
----------

Pass the initial population, generator, and helper metadata to the runner:

.. code-block:: python

    result = smc.run(
        initial_state,
        rng=rng,
        shared_context={
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "mean": mean,
            "sigma": sigma,
        },
        metadata=metadata,
    )

The runner makes ``shared_context`` available as ``context.shared`` in both
target functions. The Gaussian example supplies the bounds, mean, and standard
deviation shown above; the Rosenbrock example supplies bounds and its two
coefficients. Treat this mapping as read-only during the run.

The helper's ``metadata`` is available as ``context.metadata`` and carries
evolving information such as the tempering parameter :math:`\beta`
(``context.metadata["beta"]``). Sequince's tempered posterior applies
:math:`\beta` to the likelihood; the likelihood functions return their ordinary
log likelihood, without applying :math:`\beta` themselves.
``context.t`` identifies the current SMC iteration. The configured sampler
stops when it reaches the posterior. Logging shows progress and the running
evidence estimate.

See "Meta data & shared context" in the main Getting started guide and
"Working with particle state" in the User guide for these interfaces.

5. Inspect and save the result
------------------------------

``result.final_state`` contains the final particles, their weights, and the log
evidence estimate. ``result.weighted_mean()`` and
``result.weighted_covariance()`` compute posterior summaries using those weights.
For the Gaussian, compare the summaries and evidence with the known values.
Monte Carlo estimates fluctuate; agreement improves with adequate particles and
mixing, rather than being exact for a single run.

Each script saves:

* ``corner.png``: marginal and joint posterior plots.
* ``history.png``: the tempering parameter and log evidence over iterations.
* ``posterior.npz``: particles, normalized weights, and the log evidence estimate.
* ``result.h5``: the SMC result saved using ``result.save_to_file``.

The arrays can be loaded with ``numpy.load``. Retain the weights when analyzing
the particles. See the main API reference for ``SMCResult`` for posterior sample
extraction, diagnostics, plotting, and result persistence.

6. Explore a curved posterior
------------------------------

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

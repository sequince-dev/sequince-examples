NumPy backend
=============

This example samples the Himmelblau target described in the
:doc:`array backends overview </array_backends>`.

Start with ``run_numpy.py``. Its callbacks use NumPy operations on batches
of particles. The likelihood returns one value per particle:

.. code-block:: python

   def log_likelihood(particles, context):
       x, y = particles[:, 0], particles[:, 1]
       return -((x**2 + y - 11.0) ** 2 + (x + y**2 - 7.0) ** 2)

Initialize the population by drawing from the box prior. ``ParticleState``
supplies equal initial weights, appropriate for these prior draws:

.. code-block:: python

   rng = np.random.default_rng(seed)
   particles = rng.uniform(
       lower_bound, upper_bound, size=(n_particles, n_dimensions)
   )
   initial_state = ParticleState(particles=particles)

Configure tempering and DE, then run the sampler with the same generator:

.. code-block:: python

   smc, _, metadata = make_standard_bayesian_smc(
       log_prior=log_prior,
       log_likelihood=log_likelihood,
       target_ess_ratio=0.7,
       always_resample=True,
       mutation="differential-evolution",
       mutation_n_steps=10,
       particle_transform="affine",
   )
   result = smc.run(initial_state, rng=rng, metadata=metadata)

The prior callback includes the box normalization and returns negative infinity
outside the bounds. To adapt this example, replace the callbacks, dimensions,
bounds, and initial draws together. Each callback accepts a particle batch and
returns one log density per particle. Keep the prior normalized when estimating
evidence.

Running the example
-------------------

Run from ``examples/array_backends`` with the repository's ``default``
Pixi environment, or another environment containing Sequince and its plotting
extras::

   python run_numpy.py

The default output directory is ``outdir/numpy``.

Edit ``n_particles``, ``seed``, and ``outdir`` at the top of the script. See the
:doc:`overview </array_backends>` for shared settings and guidance on
interpreting the plots.

JAX backend
===========

This example samples the Himmelblau target described in the
:doc:`array backends overview </array_backends>`.

In ``run_jax.py``, the callbacks use ``jax.numpy``. The script runs DE and
BlackJAX NUTS in turn, starting each run from the same seed. Both use
the same functional execution path. Split the initial key so prior
initialization and sampling use separate random draws:

.. code-block:: python

   particle_key, random_state = jax.random.split(jax.random.key(seed))
   initial_state = ParticleState(
       particles=jax.random.uniform(
           particle_key,
           shape=(n_particles, n_dimensions),
           minval=lower_bound,
           maxval=upper_bound,
       )
   )

The DE configuration matches the eager examples. For BlackJAX, configure
NUTS through the sampler's mutation options:

.. code-block:: python

   mutation = "blackjax"
   mutation_kwargs = {
       "algorithm": "nuts",
       "algorithm_kwargs": {
           "step_size": 0.02,
           "inverse_mass_matrix": jnp.ones(n_dimensions),
       },
   }

Pass these as ``mutation=mutation`` and ``mutation_kwargs=mutation_kwargs``
to ``make_standard_bayesian_smc``, keeping ``mutation_n_steps=10``. For NUTS,
this means ten transitions per particle per tempering stage; each transition
can involve several leapfrog steps. NUTS uses gradients of the log target,
whereas DE uses differences between complementary particles. These runs do
not have equal computational costs.
The JAX script prints ``Total likelihood evaluations`` for either method,
using ``result.evaluation_counts.log_likelihood_evaluations`` to compare
their recorded likelihood work.

BlackJAX evaluates and differentiates one particle at a time. The JAX
callbacks therefore use ``particles[..., 0]`` and ``particles[..., 1]`` so
they accept both an individual vector and a particle batch.

The NUTS step size and inverse mass matrix are fixed in the affine-transformed
coordinates; this example does not run BlackJAX warm-up or tune them. The box
prior has a hard boundary, and NUTS moves are local, so inspect mode coverage
across seeds rather than assuming it will move between separated peaks.

For either choice, set ``logging_frequency=0`` to keep Python logging outside
the compiled iteration. ``compile_functional_iteration`` uses JAX to compile
the weighting, resampling, and mutation steps of one SMC iteration together.
Passing the resulting callable to ``run_functional`` reuses that compiled
computation at each tempering stage, reducing Python overhead within the
iteration. The outer loop stays in Python, where Sequince checks when to
stop and collects results.

The helper initializes the functional state and context before compiling,
and checks that the iteration preserves their structure, shapes, and dtypes.
Compilation does not consume a sampling transition. Compile the iteration,
then execute it through the functional API:

.. code-block:: python

   iteration = compile_functional_iteration(
       smc, initial_state, random_state, metadata=metadata
   )
   result, final_random_state = smc.run_functional(
       initial_state,
       random_state=random_state,
       metadata=metadata,
       iteration=iteration,
   )
   jax.block_until_ready(result.final_state.particles)

JAX can dispatch computations asynchronously: Python may receive an array
before the device has finished computing its values.
``jax.block_until_ready(result.final_state.particles)`` waits for the final
particle array to be ready before the script reports results and plots them.
It synchronizes with the device without copying the particles to NumPy.
If you time sampling, put the end of the timing interval after this call so
the measurement includes device execution rather than only dispatch. See
JAX's `asynchronous dispatch guide
<https://docs.jax.dev/en/latest/async_dispatch.html>`_ for more detail.

Retain ``final_random_state`` for subsequent random operations. The script
reports the device used by its final particles.

Running the example
-------------------

Run from ``examples/array_backends`` with the repository's ``jax`` Pixi
environment, which includes JAX on CPU and BlackJAX::

   python run_jax.py

For another Python environment, install Sequince's ``blackjax`` extra to run
both methods. To run only DE, remove ``"blackjax"`` from the loop in ``main``.

The script selects CPU with ``platform = "cpu"`` at the top of the file.
For a GPU run, use an environment with GPU-enabled JAX and change that line
to ``platform = "cuda"``. The script applies this setting through
``jax.config.update("jax_platforms", platform)``, overriding the
repository environment's ``JAX_PLATFORMS=cpu`` selection.

Outputs go to ``outdir/jax/differential-evolution`` and
``outdir/jax/blackjax``, so the plots can be compared without overwriting one
another.

Edit ``n_particles``, ``seed``, and ``outdir`` at the top of the script. See the
:doc:`overview </array_backends>` for shared settings and guidance on
interpreting the plots.

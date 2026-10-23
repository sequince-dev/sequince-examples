Sequince examples
=================

Start with the basic examples to learn how to define a prior and likelihood,
initialize particles, run Sequential Monte Carlo (SMC), and interpret weighted
posterior samples and evidence estimates. Work through the Gaussian target
first, then try the curved Rosenbrock posterior.

.. toctree::
   :maxdepth: 1
   :caption: Getting started

   _generated/examples/basic_examples/gaussian
   _generated/examples/basic_examples/rosenbrock

Once you are comfortable with that workflow, explore ways to adapt mutation,
change the coordinates used by the sampler, and reuse particle populations.
Each example has its own tutorial page with explanations and script source.
Example plots appear directly on that page.

.. toctree::
   :maxdepth: 1
   :caption: Sampling methods

   _generated/examples/adaptive_mutation_steps/README
   _generated/examples/preconditioning/README
   _generated/examples/persistent_sampling/README
   _generated/examples/waste_free_smc/README

For sequential observations, the particle-filtering example compares bootstrap
and observation-conditioned proposals with a Kalman-filter reference.

.. toctree::
   :maxdepth: 1
   :caption: Sequential data

   _generated/examples/particle_filtering/README

The basic, adaptive mutation, flow preconditioning, persistent sampling, and
waste-free SMC, and particle-filtering tutorials include plots alongside their
explanations. Their
run metadata identifies the source revision, parameters, seed, and pinned
dependencies.

For background and the full interface reference, see Sequince's
`user guide <https://sequince.readthedocs.io/en/latest/user_guide.html>`_
and `API reference <https://sequince.readthedocs.io/en/latest/api/index.html>`_.
The examples use :external:py:class:`~sequince.state.ParticleState` to represent the
weighted particle population.

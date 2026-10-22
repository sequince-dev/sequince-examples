PyTorch backend
===============

This example samples the Himmelblau target described in the
:doc:`array backends overview </array_backends>`.

In ``run_torch.py``, the callbacks use ``torch`` operations. The random
generator selects the device for both initialization and later random draws:

.. code-block:: python

   rng = RandomGenerator(
       backend="torch", seed=seed, device=device
   )
   particles = rng.uniform(
       lower_bound, upper_bound, size=(n_particles, n_dimensions)
   )
   initial_state = ParticleState(particles=particles)

Set ``device = "cpu"`` or, with a CUDA-enabled installation,
``device = "cuda"`` at the top of the script.
Array constructors such as ``torch.full_like`` in the prior callback preserve
the particles' device. This example uses eager execution.

Running the example
-------------------

Run from ``examples/array_backends`` with the repository's ``flow``
Pixi environment, which provides PyTorch on CPU::

   python run_torch.py

With a CUDA-enabled PyTorch installation, change the ``device`` setting
to ``"cuda"`` before running the same command.

The default output directory is ``outdir/torch``.

Edit ``n_particles``, ``seed``, and ``outdir`` at the top of the script. See the
:doc:`overview </array_backends>` for shared settings and guidance on
interpreting the plots.

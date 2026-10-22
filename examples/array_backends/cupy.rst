CuPy backend
============

This example samples the Himmelblau target described in the
:doc:`array backends overview </array_backends>`.

.. note::

   This example requires an NVIDIA GPU with a working CUDA driver and a
   compatible CuPy installation. It cannot run on CPU; use the
   :doc:`NumPy example </array_backends/numpy>` for CPU sampling.

In ``run_cupy.py``, the prior uses ``cupy`` operations and ``orng`` draws
the initial particles on CUDA:

.. code-block:: python

   rng = RandomGenerator(backend="cupy", seed=seed)
   particles = rng.uniform(
       lower_bound, upper_bound, size=(n_particles, n_dimensions)
   )
   initial_state = ParticleState(particles=particles)

The sampler configuration and eager run call follow the
:doc:`NumPy example </array_backends/numpy>`.
Keep model calculations on the CuPy arrays; copying particles to the host
inside a callback would introduce device transfers during sampling.

Running the example
-------------------

Run from ``examples/array_backends`` in a Python environment containing
Sequince, its plotting extras, and CuPy compatible with the installed CUDA
runtime::

   python run_cupy.py

The default output directory is ``outdir/cupy``.

Edit ``n_particles``, ``seed``, and ``outdir`` at the top of the script. See the
:doc:`overview </array_backends>` for shared settings and guidance on
interpreting the plots.

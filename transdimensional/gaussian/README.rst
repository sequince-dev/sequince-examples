Gaussian dimension expansion
============================

`run_transdimensional_gaussian.py <run_transdimensional_gaussian.py>`_
adds coordinates to a standard Gaussian likelihood with independent
uniform priors. It starts with zero coordinates and expands to four by
default, using 5,000 particles and random seed 1235.

Run from this directory (``transdimensional/gaussian``), using an environment
with Sequince, NumPy, MiniPCN, Matplotlib and corner installed:

.. code-block:: bash

   python run_transdimensional_gaussian.py

Expanding the target
--------------------

``GaussianExpansionTransformation`` draws one new coordinate uniformly on
``[-exp(3) / 2, exp(3) / 2]``. Its inverse drops the last coordinate and its
Jacobian is one. ``ExpandDimensions`` updates ``n_dims`` and prepares the
bridge for the enlarged model.

The expansion proposal contains the previous coordinates' likelihood and
the new coordinate's prior. These cancel in the target-to-proposal ratio,
leaving just the new coordinate's normalized Gaussian likelihood:

.. literalinclude:: run_transdimensional_gaussian.py
   :language: python
   :pyobject: log_density_ratio

The inner sampler chooses temperature increments targeting an effective
sample size of 90% of the population, resamples, and applies 50 MiniPCN
mutation steps using ``step_fn="tpcn"`` and an ``AffineTransform``.
The outer loop stops when ``target_dims`` is reached.

Reading the results
-------------------

The script prints an inner-run summary and a reference log evidence of
``-3 * target_dims``. This reference neglects Gaussian mass outside the
prior bounds; the bounds are about ten standard deviations from zero, so
the correction is tiny. Posterior coordinates should have means near zero
and variances near one.

The final posterior corner plot is saved as
``transdimensional_gaussian_corner.png`` in the working directory.
To change the run size or disable plotting, call ``main()`` from Python:

.. code-block:: python

   from run_transdimensional_gaussian import main

   result = main(
       n_particles=500, target_dims=3, mutation_steps=10, plot=False
   )

There are no command-line configuration options. The returned object holds
the completed inner runs and their final states.

.. literalinclude:: run_transdimensional_gaussian.py
   :language: python
   :pyobject: main

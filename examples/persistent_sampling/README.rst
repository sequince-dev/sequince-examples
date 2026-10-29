Persistent sampling
===================

This example demonstrates how to use persistent sampling in ``sequince`` to
sample an n-dimensional Rosenbrock target.

Persistent sampling is a variation of SMC proposed by
`Karamanis and Seljak (2025)`_ in *Persistent sampling: enhancing the efficiency
of sequential Monte Carlo*. It retains and reweights populations from previous
stages.

Run the example from the repository root:

.. code-block:: bash

   python persistent_sampling/rosenbrock.py


For a larger run:

.. code-block:: bash

   python persistent_sampling/rosenbrock.py \
       --dimensions 16 \
       --n-particles 1024 \
       --target-ess 2048 \
       --target-total-ess 8192 \
       --mutation-steps 32


The script prints diagnostics from the final weighted persistent state,
including its archive size, effective sample size, weighted mean, best
Rosenbrock potential, and log evidence. Unless ``--no-plot`` is supplied, it also
saves a weighted corner plot as ``examples/persistent_sampling/rosenbrock_<D>d.png``.
Use ``--output PATH`` to choose a different location.

Run the following for all available options:

.. code-block:: bash

   python persistent_sampling/rosenbrock.py --help


Preconditioning
---------------

First read :doc:`../preconditioning/README` for an explicit flow-transform
setup. This example also shows how to use the normalizing-flow-based preconditioning
described in `Karamanis et al. (2022)`_, *Accelerating astronomical and
cosmological inference with Preconditioned Monte Carlo*, combined with
persistent sampling from `Karamanis and Seljak (2025)`_.

The ``--flow`` option enables learned normalizing-flow preconditioning:

.. code-block:: bash

   python persistent_sampling/rosenbrock.py --dimensions 6 --flow


The flow option requires ``sequince-density`` and its flow dependencies.

.. _Karamanis et al. (2022): https://doi.org/10.1093/mnras/stac2272
.. _Karamanis and Seljak (2025): https://doi.org/10.1007/s11222-025-10682-y

Example plots
-------------

.. figure:: /_generated/results/persistent-sampling/corner.png
   :alt: Posterior marginals and joint distributions from the persistent Rosenbrock run, using the script defaults.

   Posterior marginals and joint distributions from the persistent Rosenbrock run, using the script defaults.

:download:`Run provenance </_generated/results/persistent-sampling/metadata.json>`.

Complete script
---------------

:download:`Download the script <rosenbrock.py>`.

.. literalinclude:: rosenbrock.py
   :language: python
   :linenos:

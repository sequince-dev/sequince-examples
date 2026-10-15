Particle filtering with Sequince
================================

This example shows how to implement particle filters in Sequince.

It compares two particle filters for a scalar linear-Gaussian
state-space model:

- a bootstrap filter, which proposes from the transition model;
- a locally optimal filter, which proposes conditional on the current
  observation.

Both filters process the same simulated observations. Their filtering means,
variances, and log evidence are compared with the exact Kalman-filter result.

Running the example
-------------------

.. code-block:: bash

   python particle_filter.py


The script prints accuracy, effective-sample-size, and evidence diagnostics,
then writes ``particle_filter.png`` in this directory.

Useful options include:

.. code-block:: bash

   python particle_filter.py --particles 10000 --observations 30
   python particle_filter.py --ess-threshold 0.5
   python particle_filter.py --output /tmp/particle-filter.png


By default both filters resample after every observation. Passing
``--ess-threshold FRACTION`` instead resamples only when the effective sample
size divided by the particle count falls below that fraction.

Model
-----

The example uses

.. code-block:: text

   x_0 ~ Normal(0, sigma_0^2)
   x_t | x_(t-1) ~ Normal(a x_(t-1), sigma_x^2)
   y_t | x_t ~ Normal(x_t, sigma_y^2).


The model is linear and Gaussian, so a Kalman filter supplies exact filtering
moments and log evidence for comparison.

Why the filters use different steps
-----------------------------------

For a proposal distribution ``q``, particle filtering applies the incremental
weight

.. code-block:: text

   p(y_t | x_t) p(x_t | x_(t-1)) / q(x_t | x_(t-1), y_t).


In the bootstrap filter, ``q`` is the transition distribution itself. The
transition and proposal terms therefore cancel, leaving only the observation
likelihood. A regular propagation step followed by ``IncrementalWeighting`` is
enough: after propagation, the weighting step only needs the new particles.

The locally optimal proposal is not the transition distribution, so its
weights also need the transition-to-proposal correction. That correction
depends on both the parent particles and their proposed children. A standalone
propagation step replaces the particles before the next step runs, so an
ordinary ``IncrementalWeighting`` step would no longer have both values.
``ImportanceTransition`` instead samples the children and computes the complete
correction while parent and child particles are simultaneously available.

The remaining steps are identical. Both filters record the weighted state and
then resample; only proposal generation and its associated weighting differ.

Bootstrap filter
----------------

The bootstrap filter is assembled from this Sequince pipeline:

.. code-block:: text

   PropagationStep
   -> IncrementalWeighting
   -> RecordFilteringState
   -> SystematicResampling


``PropagationStep`` samples from ``p(x_t | x_(t-1))``, and
``IncrementalWeighting`` applies ``log p(y_t | x_t)``. The local
``RecordFilteringState`` step saves a detached copy of the weighted population
before resampling. These pre-resampling populations give the filtering
moments; ``SMCResult.states`` contains the post-resampling populations passed to
the next iteration.

Locally optimal proposal filter
-------------------------------

The second filter samples from
``q(x_t | x_(t-1), y_t) = p(x_t | x_(t-1), y_t)``. The proposal parameters are

.. code-block:: text

   V_q = (1 / sigma_x^2 + 1 / sigma_y^2)^-1
   m_q = V_q (a x_(t-1) / sigma_x^2 + y_t / sigma_y^2).


Its pipeline is

.. code-block:: text

   ImportanceTransition(OptimalProposalKernel)
   -> RecordFilteringState
   -> SystematicResampling


The proposal kernel sees parent and child particles together and returns the
complete importance increment

.. code-block:: text

   log p(x_t | x_(t-1)) - log q(x_t | x_(t-1), y_t)
   + log p(y_t | x_t).


``ImportanceTransition`` applies that increment, normalizes the weights, and
updates the log evidence. Keeping the proposal and its correction together
avoids storing an extra parent population in each state.

The lower panel of the output figure compares effective sample size before
resampling. Conditioning on the observation generally gives the locally
optimal proposal a higher ESS than the bootstrap proposal.

Example plot
------------

.. figure:: /_generated/results/particle-filtering/particle_filter.png
   :alt: Bootstrap and locally optimal particle filters compared with the exact Kalman filter, including effective sample size before resampling.

   Bootstrap and locally optimal particle filters compared with the exact
   Kalman filter. The lower panel shows effective sample size before resampling;
   read it alongside filtering accuracy and the evidence diagnostics.

:download:`Run provenance </_generated/results/particle-filtering/metadata.json>`.

Complete script
---------------

:download:`Download the script <particle_filter.py>`.

.. literalinclude:: particle_filter.py
   :language: python
   :linenos:

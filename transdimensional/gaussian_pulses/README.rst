Gaussian pulse expansion
========================

`gaussian_pulses.py <gaussian_pulses.py>`_ fits a sum of Gaussian pulses to
noisy observations. The default run generates four pulses on 1,001 points
between -5 and 5, then fits models with one through six pulses using
200 particles and random seed 1751.

Run from this directory (``transdimensional/gaussian_pulses``), using an
environment with Sequince, NumPy, Matplotlib and corner installed:

.. code-block:: bash

   python gaussian_pulses.py
   python gaussian_pulses.py --true-pulses 8 --max-pulses 12

Model and expansion
-------------------

Each pulse has the form ``A * exp(-0.5 * ((x - mu) / width)**2)``.
Particles store interleaved amplitude, location and width triples, with
independent uniform priors on ``[0.001, 1.5]``, ``[-4, 4]`` and ``[0.1, 1]``
respectively. The likelihood includes the Gaussian normalization and
assumes independent measurement noise with known standard deviation 0.1.

The first inner run tempers the one-pulse prior to its posterior.
``PulseAdditionTransformation`` then appends a pulse drawn from its prior,
retains the old pulse coordinates and resets the temperature to zero.
The inverse removes the new triple and the Jacobian is one.
The new pulse's prior cancels against its proposal density, leaving the
new-minus-old log likelihood for each expansion bridge:

.. literalinclude:: gaussian_pulses.py
   :language: python
   :pyobject: log_density_ratio

The inner sampler targets an effective sample size of 60%, resamples, and
applies 20 differential evolution mutation steps by default. Each count
is fully tempered before expansion. Model comparisons assume equal prior
probabilities for the candidate pulse counts.

Reading the results
-------------------

The script prints the log evidence for each completed pulse count and
selects the count with the largest estimate. Evidence differences are log
Bayes factors and, under the equal model priors, log posterior model odds.
Finite particle populations and limited mutation can affect these
estimates; the selected count need not match the generating count.

Figures are written to ``outdir/<true_pulses>_pulses_prior/``:

- ``data.png`` compares noisy observations with the generating signal.
- ``temperature_schedule.png`` shows temperatures and pulse counts.
- ``evidence.png`` compares completed models' log evidence estimates.
- ``fit.png`` shows the selected model's weighted predictive mean and
  pointwise 90% predictive interval, including measurement noise.
- ``pulse_parameters.png`` tracks labelled posterior parameter means,
  marking introductions and true values. Labels may swap during mutation.
- ``posterior_corner.png`` overlays labelled pulse marginals for the
  selected model and marks all true pulse parameters without matching labels.

For a smaller run without figures, call ``main()`` from Python:

.. code-block:: python

   from gaussian_pulses import main

   result = main(
       true_pulses=2, max_pulses=3, n_particles=64,
       mutation_steps=2, n_observations=101, plot=False,
   )

``main()`` returns the recorded inner results. Its ``outdir`` keyword sets
the parent output directory when plotting is enabled.

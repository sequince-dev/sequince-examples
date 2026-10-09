Transdimensional SMC
====================

For the methodology of using transformations to move SMC particles between
models of different dimensions, see `Everitt et al. (2020)`_,
*Sequential Monte Carlo with transformations*.

These examples use sequential Monte Carlo (SMC) to move through models of
increasing dimension. An outer sampler expands the particle coordinates,
then ``MetaSMCStep`` runs an inner sampler that adaptively tempers towards
the new model's posterior. Each bridge reaches temperature one before the
next expansion resets the temperature to zero.

Start with the Gaussian example to see how to add a single coordinate.
The Gaussian mixture and Gaussian pulse examples add model components and
compare results across component counts. The model sizes are visited in a
fixed increasing sequence; particles do not independently choose their
dimension or make birth and death moves.

Expansion proposals
-------------------

An ``SMCTransformation`` defines the expansion, its inverse, the density of
the auxiliary draws and the Jacobian. Passing the posterior to the
transformation step lets the next tempering bridge start from the
transformed previous target. Drawing new parameters from their priors
allows those prior terms to cancel in the bridge's density ratio.
Splitting a component instead requires the split proposal density and
Jacobian, as demonstrated by the mixture example.

The inner sampler uses ``AdaptiveTemperedWeighting``, systematic resampling
and mutation. ``record_outputs=True`` retains the inner results, available
through ``result.get_meta_results("inner_results")``. Use each completed
inner run's final state for its model's posterior and evidence estimate;
intermediate states belong to tempering bridges.

Run each example from its own directory. The pages below describe the
dependencies, configuration and outputs. The mixture example includes a
Poisson prior on component count in its comparison scores; the pulse
example assumes equal prior probabilities for the candidate counts.

.. toctree::
   :maxdepth: 1

   gaussian/README
   gmm/README
   gaussian_pulses/README

.. _Everitt et al. (2020): https://arxiv.org/abs/1612.06468

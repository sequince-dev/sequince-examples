Gaussian mixture expansion
==========================

`gmm_transdimensional.py <gmm_transdimensional.py>`_ fits a one-dimensional
Gaussian mixture by successively adding components. The default run draws
400 observations from three components, then fits models with one through
six components using 500 particles and random seed 2048. Every fitted
component has its own inferred weight, mean and standard deviation.

Run from this directory (``transdimensional/gmm``), using an environment
with Sequince, NumPy, SciPy and Matplotlib installed:

.. code-block:: bash

   python gmm_transdimensional.py
   python gmm_transdimensional.py --expansion split --max-components 6
   python gmm_transdimensional.py --expansion split-random --true-components 4

``--target-components`` is an alias for ``--max-components``. Other settings
are keyword arguments to ``main()``, including ``n_particles``, ``n_samples``,
``mutation_steps`` and ``plot``.

Model and priors
----------------

For K components, particles contain K logits, K means and K log standard
deviations, in that order. A softmax converts logits to mixture weights.
Logits have independent normal priors with standard deviation 2; log widths
have independent normal priors with mean ``log(0.75)`` and standard deviation
0.75. Means have independent uniform priors between the observed extremes
extended by three times the largest data-generating standard deviation.
These bounds are set from the simulated dataset and held fixed across counts.

``log_prior`` also includes a Poisson component-count weight with rate 3.
Components retain their labels without sorting. Label switching can make
averaged component parameters hard to interpret; the posterior predictive
density instead averages each particle's mixture density using its weight.

Adding components
-----------------

The first inner run tempers from the one-component prior to its posterior.
Each later run starts from an expansion of the completed previous model:

- ``prior`` draws the new logit, mean and log width from their priors.
  Their densities cancel in the bridge ratio, leaving the new-minus-old
  log likelihood and ``log(3 / new_count)`` from the count prior.
- ``split`` splits the last labelled component. It preserves its total
  weight and weighted mean, while perturbing the child log widths.
  The split does not preserve variance; the forward absolute Jacobian is 2.
- ``split-random`` chooses a parent uniformly for each particle and
  evaluates a proposal marginalized over all possible parent choices.

The split strategies use the full transformed proposal density for
weighting. Each inner sampler targets an effective sample size of 60%,
resamples, and applies 10 differential evolution mutation steps by default.

.. literalinclude:: gmm_transdimensional.py
   :language: python
   :pyobject: GMMSplitTransformation

Reading the results
-------------------

The script prints parameter summaries for the final, largest model,
the posterior predictive density error against the generating mixture,
and scores for all completed component counts. The final density plot
also uses the largest model, even if another count has the highest score.

The reported scores include relative Poisson count-prior weights. With
``Z_K`` denoting conditional model evidence and ``p(K)`` the count prior,
the accumulated score estimates ``log(Z_K) + log(p(K) / p(1))``.
Consequently, differences printed as ``ln BF`` estimate log posterior
model odds against one component. To recover a log Bayes factor, subtract
``log(p(K) / p(1))`` from that difference. The count-prior normalization
over the visited models cancels in these comparisons.

Two figures are saved in the working directory, with the expansion name
substituted for ``<expansion>``:

- ``gmm_transdimensional_<expansion>.png`` shows the final predictive density,
  component counts, temperatures and temperature increments.
- ``gmm_component_means_<expansion>.png`` tracks labelled posterior mean
  locations, marking when components are introduced. Labels may swap.

Set ``plot=False`` when calling ``main()`` to return the inner results
without writing figures.

Flow preconditioning
====================

``sequince`` supports preconditioning for mutation steps, as described by
`Karamanis et al. (2022)`_. A normalizing flow can learn a transformation from
the current particle population to a simpler latent space, where the mutation
kernel can explore more efficiently.

``sequince.transforms.FlowTransform`` adapts a normalizing flow for use as a
particle transform. The flow must implement the
``sequince.flows.NormalizingFlow`` protocol:

Construct the transform by passing it a compatible flow and, optionally, the
keyword arguments used to fit that flow:

.. code-block:: python

   from sequince.transforms import FlowTransform

   flow = ...
   preconditioning_transform = FlowTransform(
       flow=flow,
       fit_kwargs={...},
   )

The resulting object can be used anywhere a particle transform is accepted by
``sequince``.

Fitting the flow
----------------

The fitting of the transform is handled by the mutation step. The frequency
of fitting can be controlled by the ``transform_refit`` argument.
Current options are:

- ``"always"``: fit the flow every time the mutation step is called (default)
- ``"once"``: fit the first time the mutation step is called, then reuse
- ``"per-context"``: fit once per unique context, then reuse for that context

When using the prebuilt samplers, the ``transform_refit`` argument should be
passed in the ``mutation_kwargs`` dictionary.

Prebuilt flows
--------------

``sequince-density`` provides various normalizing flows that can be used for
preconditioning via the ``zuko`` and ``flowjax`` packages. See the
``sequince_density.flows.zuko`` and ``sequince_density.flows.flowjax`` modules
for more information.

Continue with :doc:`../persistent_sampling/README` to combine this approach
with persistent sampling.

Example
-------

The following example uses a Zuko masked autoregressive flow to precondition a
four-dimensional Rosenbrock target. It requires ``sequince-density`` with its
Zuko backend and PyTorch support.

Example plots
-------------

.. figure:: /_generated/results/flow-preconditioning/rosenbrock_corner.png
   :alt: Posterior marginals and joint distributions for the four-dimensional Rosenbrock target sampled with flow preconditioning.

   Posterior marginals and joint distributions for the four-dimensional Rosenbrock target sampled with flow preconditioning.

:download:`Run provenance </_generated/results/flow-preconditioning/metadata.json>`.

:download:`Download the script <flow_preconditioning.py>`.

.. literalinclude:: flow_preconditioning.py
   :language: python
   :linenos:

.. _Karamanis et al. (2022): https://doi.org/10.1093/mnras/stac2272

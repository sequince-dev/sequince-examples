Array backends
==============

Using array backends
--------------------

Sequince can sample with NumPy, CuPy, PyTorch, and JAX arrays. The particle
arrays select the numerical backend, and the model callbacks must evaluate
their log densities using compatible array operations. NumPy runs on CPU;
CuPy requires CUDA; PyTorch and JAX can run on CPU or supported accelerators.
Backend support also depends on the mutation method and other components you
choose. The standard differential-evolution (DE) sampler used here supports
all four backends.

Eager and functional execution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sequince provides eager and functional ways to execute a sampler.

In the eager API, call ``smc.run(initial_state, rng=rng, metadata=metadata)``.
The sampler advances a mutable random generator as it applies weighting,
resampling, and mutation steps. The NumPy, CuPy, and PyTorch examples use this
approach.

In the functional API, call ``smc.run_functional`` with an explicit random
state. It returns both the sampling result and the updated random state.
The JAX example uses ``compile_functional_iteration`` to compile one SMC
iteration and passes that iteration to ``run_functional``. The outer sampling
loop remains under host control. Callbacks in the compiled iteration must be
JAX-traceable; avoid converting their arrays to NumPy or Python scalars.

Random number generation with orng
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sequince uses ``orng`` to provide a common random-number interface across
array libraries. Its ``RandomGenerator`` exposes NumPy-style methods such as
``uniform``, ``normal``, and ``choice``, while drawing with the selected
backend. It is installed as a dependency of Sequince.

The CuPy and PyTorch examples construct this generator explicitly::

   from orng import RandomGenerator

   rng = RandomGenerator(backend="cupy", seed=42)
   particles = rng.uniform(-6.0, 6.0, size=(1000, 2))

Here ``particles`` is a CuPy array on the GPU. With ``backend="torch"``,
draws are PyTorch tensors; pass ``device="cpu"`` or ``device="cuda"`` to
choose their device. Pass the same generator to ``smc.run(..., rng=rng)``
so initialization, resampling, and DE proposals advance one random stream.
The generator is mutable: drawing samples changes its internal state.

The NumPy example uses ``np.random.default_rng(seed)`` directly. Sequince
wraps this existing generator in ``orng`` internally, retaining its state
after the initial prior draws. You can also use
``RandomGenerator(backend="numpy", seed=seed)`` explicitly.

The JAX functional example supplies a JAX random key rather than a mutable
generator. Sequince's steps use ``orng`` internally to obtain backend random
draws and return the updated key through the functional API. Keep using the
returned key, rather than reusing an earlier key, for subsequent draws.
Equal integer seeds across libraries do not imply identical random streams.

Writing portable array code
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``array_api_compat.array_namespace(particles)`` to obtain a compatible
namespace, conventionally named ``xp``, from an input array. Call operations
such as ``xp.all`` and ``xp.where`` through that namespace to keep results
backend-native. Constructors such as ``xp.full_like`` preserve an existing
array's dtype and device. This lets you share model code without depending
on Sequince-specific array helpers.

``array-api-extra`` (imported as ``array_api_extra``) complements the standard
namespace with portable helpers for operations outside the Array API standard.
For example, ``array_api_extra.atleast_nd(particles, ndim=2, xp=xp)`` prepends
axes when needed to give an array at least two dimensions. Its ``at`` helper
also provides indexed updates compatible with immutable JAX arrays. Use these
helpers when the standard namespace lacks an operation you need, passing the
namespace from ``array_namespace`` through ``xp``.

For models that need data-dependent branches or loops, ``acai`` complements
``array_api_compat`` with ``cond``, ``while_loop``, ``fori_loop``, and ``scan``.
For example, a Python ``if`` on an array value cannot be traced by JAX;
``acai.cond`` dispatches to ``jax.lax.cond`` for JAX and uses eager Python
control flow for NumPy, CuPy, and PyTorch. Both JAX branches must be traceable
and return compatible structures, shapes, and dtypes. PyTorch execution is
eager here; this does not guarantee compatibility with ``torch.compile``.
Sequince uses ``acai`` internally for its functional control flow.

``array_api_compat`` supplies compatible array namespaces, ``array_api_extra``
adds portable array helpers, ``orng`` supplies random draws, and ``acai``
supplies control flow. These libraries let model code work across backends
without depending on Sequince-specific array helpers.

The Himmelblau examples
-----------------------

The following examples run the same sampling problem on each backend so you
can compare how arrays, random generators, and execution styles fit together.
Start with the NumPy script for the complete eager workflow, then compare
CuPy and PyTorch for device-specific execution and JAX for the functional API.

We use the two-dimensional Himmelblau function, an optimization benchmark
with four separated minima. Taking its negative as a log likelihood creates
a multimodal sampling problem: the sampler must explore several distinct
regions of parameter space. This synthetic target illustrates both backend
choices and the challenge of retaining multiple modes during sampling.

The prior is uniform on ``[-6, 6]`` in each coordinate. The unnormalized
likelihood is ``exp(-H(x, y))``, where

.. math::

   H(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2.

The posterior has four separated peaks, making it useful for exploring mode
coverage. By default, all examples use adaptive tempering, affine
preconditioning, and ten standard differential-evolution (DE) sweeps per
stage, selected with
``mutation="differential-evolution"``. Each sweep updates two complementary
particle groups. Each script defines its prior and likelihood using its own
backend's array operations, preserving the particles' device.
The JAX script also offers BlackJAX NUTS mutation for comparison.

Each script is self-contained, defining its target callbacks, sampler setup,
and result plotting. The backend pages highlight the parts to compare; the
scripts contain the complete runnable examples.

Running the examples
~~~~~~~~~~~~~~~~~~~~

Each backend has its own walkthrough and environment instructions:

* :doc:`NumPy </array_backends/numpy>`: the complete eager workflow on CPU.
* :doc:`CuPy </array_backends/cupy>`: eager sampling on CUDA.
* :doc:`PyTorch </array_backends/torch>`: eager sampling on CPU or CUDA.
* :doc:`JAX </array_backends/jax>`: functional execution with DE or BlackJAX NUTS.

Edit ``n_particles``, ``seed``, and ``outdir`` at the top of each script.
Defaults are 1,000 particles, seed 42, and ``outdir/<backend>`` relative to
``examples/array_backends``.
For JAX, the default output directory is ``outdir/jax/<mutation>``
(``differential-evolution`` or ``blackjax``) so the DE and BlackJAX plots can be compared without overwriting one another.
Each prints estimated log
evidence, weighted posterior means and covariance, then saves ``posterior.png``
and ``history.png``. These are manual examples;
they are not part of the automated documentation output manifest and do not
require a GPU to build their tutorial pages. Equal seeds across backends do not
produce identical random streams or particle populations.

Interpreting the results
~~~~~~~~~~~~~~~~~~~~~~~~

The four minima of the Himmelblau function lie near ``(3, 2)``,
``(-2.805, 3.131)``, ``(-3.779, -3.283)``, and ``(3.584, -1.848)``.
Inspect the corner plot for all four modes; means and covariance alone do not
describe this posterior well. Equal objective values at the minima do not imply
equal integrated posterior mass around each peak.

The history plot shows accumulated log evidence and the tempering parameter
``beta`` approaching one. The evidence refers to the stated unnormalized
likelihood and normalized box prior; no analytic evidence is supplied.
Small populations can lose modes during resampling, and ten DE sweeps do not
guarantee mixing. Increase ``n_particles`` and compare multiple seeds when
checking mode coverage.

Writing backend-agnostic model code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The backend-specific scripts can also share their target callbacks. For the
Himmelblau problem, obtain the namespace from the particles in the prior and
keep the likelihood's arithmetic unchanged:

.. code-block:: python

   from math import log

   from array_api_compat import array_namespace

   def log_prior(particles, context):
       xp = array_namespace(particles)
       inside = xp.all((particles >= -6.0) & (particles <= 6.0), axis=-1)
       density = xp.full_like(
           particles[..., 0], -particles.shape[-1] * log(12.0)
       )
       return xp.where(inside, density, xp.full_like(density, -float("inf")))

   def log_likelihood(particles, context):
       x, y = particles[..., 0], particles[..., 1]
       return -((x**2 + y - 11.0) ** 2 + (x + y**2 - 7.0) ** 2)

``array_api_compat`` provides compatible namespaces for NumPy, CuPy, and
PyTorch, and recognizes JAX's namespace. Operations through ``xp`` return
backend-native arrays. Using ``full_like`` also preserves the input dtype
and device. The Himmelblau likelihood already needs only indexing and
arithmetic, so its implementation can be shared without a namespace lookup.
These callbacks can replace the corresponding definitions in any of the
four scripts; initialization and the choice of execution API stay explicit.

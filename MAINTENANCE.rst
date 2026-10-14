Setup and maintenance
=====================

Local execution
---------------

Install Pixi 0.81.0 or newer, then run from the repository root::

   pixi install --locked -e default -e docs -e check
   pixi run --locked example gaussian
   pixi run --locked example rosenbrock
   pixi run --locked example adaptive-mutation
   pixi run --locked -e docs docs
   pixi run --locked -e check test
   pixi run --locked -e check lint
   pixi run --locked -e check format-check
   pixi run --locked -e check pre-commit run --all-files

The Makefile in ``docs/`` wraps these Pixi tasks. From the repository root::

   make -C docs setup
   make -C docs check
   make -C docs example EXAMPLE=rosenbrock
   make -C docs examples
   make -C docs docs
   make -C docs docs-only
   make -C docs validate

``make -C docs docs`` checks existing output completeness and provenance, then builds
the full Sphinx site. It reruns examples only if outputs are missing or stale.
``make -C docs validate`` runs checks, explicitly reruns all examples, then builds the
site. Both preserve execution order with ``make -C docs -j``.
You can also change into ``docs/`` and run these targets without ``-C docs``. ``make -C docs docs-only`` never
executes examples and fails if outputs are missing or stale. Use this target
when editing documentation on a machine with limited memory. Use ``make -C docs example EXAMPLE=<id> ENV=<environment>`` for
a single example requiring another backend. ``make -C docs hooks`` runs all pre-commit
hooks and may edit files. ``make -C docs help`` lists the available targets. The
``SEQUINCE_INVENTORY`` environment override described below also works with make.

``pixi run --locked examples`` runs every manifest entry using its declared
environment. Open ``docs/_build/html/index.html`` after a successful build.
The maintained narratives remain beside the scripts under ``examples/``.
The pipeline copies them and script sources into ``docs/_generated`` so relative ``literalinclude``
paths still resolve. Both that directory and ``build-docs`` are ignored.

The runner imports each script and overrides only its output-directory global.
It preserves the original working directory, parameters, sampler, seed and RNG
calls. For CLI examples the runner passes the script defaults without the
pipeline's own arguments. The three additional scripts write concise JSON
statistics only when the runner sets ``documentation_output``; ordinary script
execution keeps its existing output behavior. Matplotlib uses Agg. Raw particles and HDF5 results live in a temporary
run directory and are removed after extracting plots and weighted summaries.
Logs remain in ``build-docs``. The covariance in the summary is the weighted
population covariance; it does not apply a finite-sample correction. No target,
weighting, evidence convention or mutation setting is changed for publication.

Dependencies and new examples
-----------------------------

During development, Sequince follows the latest ``main`` branch in
``pyproject.toml``. Normal example commands refresh that dependency before
execution. ``make -C docs setup`` and ``make -C docs docs`` refresh it too;
``make -C docs docs-only`` uses the existing snapshot and never runs sampling.
Use ``make -C docs refresh`` to resolve main without executing examples.
Direct standalone Python script execution uses whichever Sequince is installed;
use the runner commands for automatic refresh.

Actions resolves main once in the manifest job, shares the resulting
``pixi.lock`` with every matrix job, and uses locked installs afterward. This
keeps one workflow internally consistent even if main advances during the run.
``SEQUINCE_LOCK_READY=1`` is reserved for such already-resolved batches or
intentional reproduction of an archived execution; it suppresses refresh.

``pixi.lock`` remains versioned as a bootstrap snapshot. Metadata verifies and
records the installed version and exact resolved Git commit. Each output bundle
includes ``execution-pixi.lock`` with checksums. RTD restores that archived lock
before validating outputs; it does not fetch today's main. The publishing job
allows this intentional lock-file change but rejects other tracked edits.

Separate executions of the same example source commit can now produce different
results as Sequince main changes. Existing immutable-by-policy bundles still
cannot be overwritten with different bytes: publish a new example source commit
for a new dependency snapshot. To reproduce an old run, use its archived lock
and ``SEQUINCE_LOCK_READY=1``. This development policy supersedes the original
requirement that dependency resolution be identical for every run of a source
commit. When Sequince releases are available, replace the branch dependency with
the latest compatible release and revise the refresh command at the same time.

The docs environment contains no sampler or optional backend. The separate
``flow`` environment uses CPU-only PyTorch and pins ``sequince-density`` to a
full Git commit in its repository (currently named ``sequince-flows``). Both Git
dependency commits and installed versions are verified and recorded for flow
runs. The separate ``jax`` environment is a CPU template for future examples.

The Sequince repository is currently private and there is no public PyPI
release. Existing SSH access was used to resolve the local lock. To use existing
SSH access locally without changing Git configuration permanently::

   GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=url.git@github.com:.insteadOf \
     GIT_CONFIG_VALUE_0=https://github.com/ pixi install --locked

Before enabling this workflow for public/fork PRs, make the dependency
publicly readable (or replace it with a compatible exact public release and
regenerate the lock). Never add a private dependency token to PR execution.
RTD only installs Sphinx and reads public output assets, so it needs no Sequince
repository access or sampling environment. The Sequince documentation inventory
at ``https://sequince.readthedocs.io/en/latest/objects.inv`` must also be publicly
readable; current probes returned HTTP 403/404 in this environment. For local
validation against an inventory generated by the main Sequince Sphinx build,
set ``SEQUINCE_INVENTORY=/absolute/path/to/objects.inv``. This overrides only the
inventory location; generated API links still point to the main Sequince docs.

Place new example directories under ``examples/`` and add a record to
``docs/examples.json`` with a unique ID, script, working
directory, environment, timeout in seconds, narrative, parameter globals,
required raw outputs and captioned figures. Scripts must expose ``main()`` and
``outdir``, save weighted particles in ``posterior.npz`` with ``particles``,
``weights`` and scalar ``log_evidence``, and list every required output.
Seed must be a declared parameter. Examples can provide a raw ``summary.json``
instead of ``posterior.npz``; the runner validates and packages it without
rendering it in the tutorial. CLI examples expose ``build_parser()`` so metadata
records their actual defaults. Optional Git dependencies list their feature
names under ``dependency_features`` in the execution manifest. Add backend features/environments to
``pyproject.toml`` and regenerate the lock as needed. The execution matrix comes from the manifest; no workflow edit is needed for
another entry. Put figure directives, captions, and summary includes directly
in the example README using ``/_generated/results/<id>/`` paths. The staging
step copies the narrative, script, and validated outputs without creating a
separate result page. Link its staged README from ``docs/index.rst`` in reading
order. Narrative-only tutorials keep their ``README.rst`` beside the script in
``examples/``. Add their directory and source filename to the staging list in
``prepare()`` and link the staged README from ``docs/index.rst``. This preserves
relative source includes without executing optional backends or adding an
execution-manifest entry. Keep contributor and hosting instructions
in this repository document rather than the tutorial navigation.

Provenance and durable bundles
------------------------------

Every example records the checked-out source SHA, hashes of its script and
narrative, manifest, runner and Pixi inputs, installed Sequince version/resolved main commit,
Python version, environment, parameters and seed. Per-example checksums are
validated before Sphinx. Local edited inputs are fingerprinted, but publication
requires a clean tracked checkout except for the resolved execution lock. Actions checks out the same SHA in each job
(including the synthetic merge commit on PRs) and downloads only same-run artifacts.

Trusted main runs create ``docs-outputs-<full SHA>.tar.gz`` and a companion
``.sha256`` asset on the dedicated prerelease ``docs-output-<full SHA>`` in
**sequince-examples**, never the library repository. The archive includes plots,
concise summaries, provenance, checksums, ``execution-pixi.lock`` and
``bundle-manifest.json``. It excludes
raw chains, checkpoints and HDF5 files. Tar and gzip headers are deterministic.
Existing assets are downloaded and compared before reuse; different bytes fail
rather than replacing archived results. Draft releases allow recovery from a
partial upload. Do not move these tags or edit release assets.

Retain release assets for as long as any corresponding RTD version may be
rebuilt. Actions artifacts expire after 14 days (logs after 7) and serve only
intermediate/PR validation. To regenerate, check out the exact source revision,
restore the bundle execution lock and set ``SEQUINCE_LOCK_READY=1`` before
executing examples, building docs and creating the bundle. A rerun
with different results intentionally fails publication: investigate the mismatch,
then publish a new source commit instead of replacing history. Hardware/BLAS
variation can change numerical or figure bytes despite a fixed seed.

Read the Docs Community setup
-----------------------------

These are manual maintainer steps; no external settings or credentials are
created by this change.

#. Make sequince-examples and its output release assets public. Import the repo
   at ``app.readthedocs.org`` as a separate Community project, with main as the
   default branch and ``.readthedocs.yaml`` as its configuration.
#. Keep ``latest`` active/default. Disable automatic push/tag builds and PR
   builds for this project. For a GitHub App integration, use automation rules
   with an enabled **Trigger build for version** rule that never matches real
   refs (for example a reserved ``^disabled-auto-build$`` ref), and remove other
   build-trigger rules. Such rules gate webhook builds; explicit API builds
   remain available. For legacy webhook integrations, disable automatic builds
   on the integration. Verify no ordinary push starts a build before enabling
   publication. Do not enable rules that activate ``docs-output-*`` tags.
#. In the main Sequince project's **Subprojects** settings attach this imported
   project with alias ``examples``. Select the examples default version as
   latest. Its URL is ``https://sequince.readthedocs.io/projects/examples/en/latest/``.
   The main Sequince build system remains unchanged. A parent navigation link to
   this URL can be added when the subproject is live.
#. Set GitHub Actions variable ``RTD_PROJECT`` to the imported project's slug
   and secret ``RTD_TOKEN`` to an RTD Community API token. Tokens inherit their
   user's permissions: use a dedicated maintainer account with access only to
   this project where possible. RTD needs no GitHub publishing token. The workflow
   uses the automatic GitHub token with ``contents: read`` for validation and
   ``contents: write`` only for the trusted publishing job. Allow Actions to
   create releases in the examples repository. No secrets enter PR jobs.
#. Merge the files, then run the workflow on main once dependency access is
   ready. Inspect the release assets and monitored RTD result. Initial import
   builds can fail until the first matching bundle is uploaded; rerun afterward.

RTD runs standard Sphinx on Community, downloading and verifying the exact bundle
for ``git rev-parse HEAD`` in ``pre_build``. It never falls back to an older
bundle. A trigger for ``latest`` does not pin a commit: the actual checkout is
verified and must have its own matching bundle. Main freshness is checked before
Sphinx and in ``post_build``, before RTD's upload.

The publishing job holds a non-cancelling concurrency lock until RTD finishes,
waits for any existing RTD builds, and checks current main before triggering.
Stale workflow runs skip triggering. A newer push during an older RTD build
causes the freshness hook to fail; its workflow then releases the lock and the
newer run can build. This prevents older workflow builds from overwriting newer
successful docs. Do not run concurrent manual builds or enable automatic builds:
RTD has no customizable upload hook or API commit-pinning guarantee. After a
monitor timeout or manual cancellation, inspect/drain RTD builds before retrying;
the next publishing job also waits for active builds.

Tagged versions and stable
--------------------------

Initially publish only main/latest. Do not automatically activate artifact tags
or a stable alias. For a curated example release, choose an immutable ``vX.Y.Z``
tag on a main commit whose output bundle already exists; sync and activate that
version in RTD, then manually build it. The hook downloads the bundle for the
tag's actual checkout. Configure stable to the newest curated release only
when adopting this policy. Never retarget a release tag; artifact prereleases
are not example releases. The current workflow intentionally does not trigger
tag/stable builds or create semantic version tags.

PRs and troubleshooting
-----------------------

PRs execute the full matrix and build Sphinx, with downloadable HTML artifacts.
They do not publish releases, receive RTD credentials or create RTD previews.
Use the HTML artifact for review. RTD PR previews would require a separate
trusted bundle/authorization policy and are disabled here.

For failed examples, inspect the named job and its log artifact, reproduce with
``pixi run --locked example <id>``, and check required filenames and the timeout.
Do not reduce particle counts or change seeds just to pass CI. For a missing
bundle, verify the full RTD checkout SHA has a successful publishing workflow
and matching release assets; execute that revision instead of selecting another
commit's outputs. A checksum/provenance failure means corruption or mixed inputs:
re-download or reproduce the exact locked revision. Missing figures fail before
Sphinx. Warnings (including missing Intersphinx inventories) fail the build.

For RTD API failures check the project slug, account permissions, token, rate
limits and build logs. A main-advanced error is expected during rapid pushes;
check that the newest workflow eventually succeeds. A private dependency error
in Actions requires public dependency access, not credentials in PR jobs.

Official integration references
-------------------------------

* `Pixi setup action and cache/locked inputs <https://github.com/prefix-dev/setup-pixi/blob/main/action.yml>`_
* `GitHub release and asset API <https://docs.github.com/en/rest/releases/releases>`_
* `RTD build hooks <https://docs.readthedocs.com/platform/stable/build-customization.html>`_
* `RTD API v3, builds and authentication <https://docs.readthedocs.com/platform/stable/api/v3.html>`_
* `RTD webhook automation rules <https://docs.readthedocs.com/platform/stable/automation-rules.html>`_
* `RTD subprojects <https://docs.readthedocs.com/platform/stable/subprojects.html>`_

Prepared plot integrations
--------------------------

The flow preconditioning, persistent sampling, and waste-free examples are now
configured for the matrix and durable bundles, with inline figure directives.
Their new integrations have not been executed or numerically validated locally.
The standalone scripts retain their particle counts, dimensions, seeds, fitting
budgets, and mutation settings. Some constants have moved to module scope so
the runner can record them. Documentation-only summary writes do not draw RNG.
Flow preconditioning uses its existing four-dimensional, 1,000-particle setup;
persistent sampling uses the parser defaults (four dimensions, 256 active
particles); waste-free SMC keeps 100 chains of length 100 and seed 123.

Do not treat pre-existing loose PNG files as verified documentation outputs.
A full build requires fresh, matching provenance for all seven execution entries.
In particular, ``make -C docs docs`` may launch sampling when outputs are stale
or missing; ``make -C docs docs-only`` never launches it. The runner has timeouts
but no enforced memory ceiling yet. Validate the new runs on suitable hardware
or after adding memory limits. Dependency resolution and mocked worker tests
do not validate numerical behavior or plot quality.

Particle filtering is also configured for automated execution, with its existing
defaults of 1,000 particles, 20 observations, seed 2026, and resampling after
every observation. The runner isolates its plot and captures accuracy, ESS,
and evidence diagnostics in the bundle without rendering numerical summaries.
This integration has mocked worker and layout checks; no filtering run was
performed while preparing it.

"""Script for running toy examples."""

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import likelihoods
import matplotlib.pyplot as plt
import numpy as np
import yaml
from sequince import ParticleState
from sequince.log import configure_logging
from sequince.posterior import draw_additional_samples
from sequince.prebuilt_samplers import make_standard_bayesian_smc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML config file for the model. If not specified, use default parameters.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    configure_logging(level="INFO")

    outdir = config.get("outdir", None)

    if outdir is None:
        outdir = (
            Path("outdir")
            / f"{config['model']['name'].lower()}_{config['model']['dimensions']}d"
        )

    # get label or use date/time if not specified
    label = config.get("label", f"run_{time.strftime('%Y%m%d-%H%M%S')}")

    # Copy the config file to the output directory for record-keeping
    outdir = Path(outdir) / label
    outdir.mkdir(parents=True, exist_ok=True)
    out_config_path = outdir / "config.yaml"
    with open(out_config_path, "w") as f:
        yaml.dump(config, f)

    ModelClass = likelihoods.models.get(config["model"]["name"].lower())
    rng = np.random.default_rng(seed=config.get("seed", 42))
    kwargs = {"xp": np}
    if config["model"]["dimensions"] is not None:
        kwargs["dimensions"] = config["model"]["dimensions"]
    if config["model"].get("arguments") is not None:
        kwargs["log_likelihood_args"] = dict(config["model"]["arguments"])
    model = ModelClass(**kwargs)

    n_particles = config["sampler"].pop("n_particles", 1000)

    initial_state = ParticleState(
        rng.uniform(
            low=model.prior_bounds[0],
            high=model.prior_bounds[1],
            size=(n_particles, model.dimensions),
        )
    )

    # Fast set-up prebuilt
    smc, posterior, metadata = make_standard_bayesian_smc(
        log_prior=model.log_prior,
        log_likelihood=model.log_likelihood,
        **config["sampler"] or {},
    )

    # run the sampler
    result = smc.run(
        initial_state,
        rng=rng,
        shared_context={},
        metadata=metadata,
        **config["run"] or {},
    )

    # Draw additional posterior samples
    print("Drawing additional posterior samples...")
    n_final_samples = config["post_processing"].pop("n_final_samples", 1000)
    posterior_state, final_result = draw_additional_samples(
        posterior=posterior,
        result=result,
        rng=rng,
        n_particles=n_final_samples,
        **config["post_processing"] or {},
    )

    # print log-evidence estimate
    print(f"Log-evidence estimate: {posterior_state.log_evidence:.2f}")
    if model.log_evidence is not None:
        print(f"True log-evidence: {model.log_evidence:.2f}")

    fig = final_result.plot_corner()
    fig.savefig(outdir / "corner_plot.png")

    # Plot the state plots
    fig, axes = plt.subplots(2, 1)
    result.plot_log_evidence(ax=axes[1])
    result.plot_metadata_history("temperature", ax=axes[0])
    fig.savefig(outdir / "history.png")

    # Dump final result and posterior state to disk for later analysis
    result.save_to_file(outdir / "final_result.h5")
    np.savez(outdir / "posterior_state.npz", **asdict(posterior_state))


if __name__ == "__main__":
    main()

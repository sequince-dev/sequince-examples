#!/usr/bin/env python
"""
Example of frequency annealing for gravitational-wave parameter estimation using
the sequince library. The example injects a binary black hole signal and
recovers it using a Sequential Monte Carlo sampler that gradually increases
the maximum frequency of the likelihood integration.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from bilby.core.sampler.base_sampler import _initialize_global_variables
from sequince.posterior import TemperedPosterior
from utils import (
    FrequencyAnnealingConfig,
    FrequencyAnnealingLikelihood,
    build_injection_parameters,
    build_interferometers,
    build_posterior_functions,
    build_priors,
    build_waveform_generator,
    configure_example_logging,
    create_initial_state,
    determine_frequency_schedule,
    get_initial_parameters,
    plot_comparison_to_reference,
    plot_intermediate_corners,
    plot_metadata_history,
    plot_posterior_corner,
    run_bilby_analysis,
    run_smc,
)

logger = logging.getLogger("sequince")

os.environ["OMP_NUM_THREADS"] = str(1)
os.environ["OPENBLAS_NUM_THREADS"] = str(1)
os.environ["MKL_NUM_THREADS"] = str(1)


def run_frequency_annealing(cfg: FrequencyAnnealingConfig) -> None:
    configure_example_logging(cfg)
    injection_parameters = build_injection_parameters(cfg)
    waveform_generator = build_waveform_generator(cfg)
    ifos = build_interferometers(cfg, injection_parameters, waveform_generator)
    priors = build_priors(cfg, injection_parameters)
    likelihood = FrequencyAnnealingLikelihood(
        interferometers=ifos,
        waveform_generator=waveform_generator,
        distance_marginalization=True,
        time_marginalization=False,
        phase_marginalization=True,
        priors=priors,
    )
    logger.info("Starting annealing run with prior keys: %s", priors.keys())

    if cfg.frequency_schedule == "auto":
        frequency_schedule = determine_frequency_schedule(
            likelihood,
            injection_parameters,
            start_frequency=20.0,
            end_frequency=cfg.sampling_frequency / 2,
            n_grid=500,
            n_steps=cfg.n_frequency_steps,
        )
        logger.info(
            "Auto-determined frequency schedule with %d bins: %s",
            len(frequency_schedule),
            frequency_schedule,
        )
    else:
        frequency_schedule = np.array(
            sorted(set(cfg.frequency_schedule)), dtype=float
        )

    initial_state = create_initial_state(
        priors, frequency_schedule, cfg.n_particles
    )

    with mp.Pool(
        processes=cfg.n_pool_processes,
        initializer=_initialize_global_variables,
        initargs=(
            likelihood,
            priors,
            priors.non_fixed_keys,
            True,
            get_initial_parameters(priors),
        ),
    ) as pool:
        log_prior, log_likelihood = build_posterior_functions(
            priors, likelihood, pool
        )
        posterior = TemperedPosterior(
            log_likelihood=log_likelihood,
            log_prior=log_prior,
        )
        result = run_smc(
            posterior,
            cfg,
            frequency_schedule,
            initial_state,
        )

        # Make a nested SMC result for the mutation steps to analyze the
        # evolution within each nested SMC run
        nested_smc_result = result.get_nested_results()

        fig, ax = plt.subplots(2, 1)
        nested_smc_result.plot_log_evidence(ax=ax[0])
        nested_smc_result.plot_metadata_history("temperature", ax=ax[1])
        fig.savefig(cfg.outdir / "nested_smc_history.png")

    for state in result.states:
        max_freq = state.metadata.get("current_maximum_frequency", 0.0)
        log_evidence = state.log_evidence
        logger.info(
            "Max frequency: %.1f Hz, log evidence: %.2f",
            max_freq,
            log_evidence,
        )

    if cfg.run_bilby_analysis:
        logger.info(
            "Running comparison bilby analyses for each frequency in the schedule..."
        )
        reference_results = run_bilby_analysis(
            cfg,
            priors,
            likelihood,
            injection_parameters,
        )

    plot_metadata_history(
        result, cfg.outdir / cfg.additional_outputs["metadata"]
    )
    plot_intermediate_corners(
        result,
        priors,
        injection_parameters,
        cfg.outdir / cfg.additional_outputs["intermediate_corner"],
    )

    plot_posterior_corner(
        result.final_state,
        priors,
        injection_parameters,
        cfg.outdir / cfg.additional_outputs["posterior_corner"],
    )

    if cfg.run_bilby_analysis:
        plot_comparison_to_reference(
            reference_results,
            result,
            priors,
            injection_parameters,
            cfg.outdir / "comparison_corner_plots",
        )

    logger.info("Finished run. Results stored in %s", cfg.outdir.resolve())


def main():
    run_frequency_annealing(
        FrequencyAnnealingConfig(
            aligned_spins=True,
            outdir=Path("frequency_annealing_metasmc"),
            n_pool_processes=8,
            n_frequency_steps=4,
            ess_ratio=0.8,
            n_particles=500,
            mutation_steps=20,
            run_bilby_analysis=True,
            bilby_sampler_kwargs=dict(
                nlive=500,
                sampler="nessai",
            ),
            fixed_keys=[
                "ra",
                "dec",
                "psi",
                "theta_jn",
                "geocent_time",
            ],
        )
    )


if __name__ == "__main__":
    main()

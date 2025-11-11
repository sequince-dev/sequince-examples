#!/usr/bin/env python
"""
Example of mode annealing for gravitational-wave parameter estimation using
the sequince library. The example injects a binary black hole signal and
recovers it using a Sequential Monte Carlo sampler that gradually adds higher
order modes to the waveform model.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
from dataclasses import dataclass, field
from itertools import repeat
from pathlib import Path
from typing import Dict, Iterable

import bilby
import corner
import matplotlib.pyplot as plt
import numpy as np
from bilby.core.sampler.base_sampler import _initialize_global_variables
from orng import ArrayRNG
from sequince.logging_utils import configure_logging
from sequince.posterior import TemperedPosterior
from sequince.smc import SequentialMonteCarlo
from sequince.state import ParticleState
from sequince.steps import (
    AdaptiveTemperedWeighting,
    MiniPCNMutation,
    SystematicResampling,
)
from sequince.steps.base import SMCStep
from sequince.steps.control import ContinueWhile
from sequince.steps.log import LoggingStep

logger = logging.getLogger("sequince")

os.environ["OMP_NUM_THREADS"] = "1"


@dataclass
class ModeAnnealingConfig:
    duration: float = 4.0
    sampling_frequency: float = 2048.0
    minimum_frequency: float = 20.0
    outdir: Path = Path("modes_more_modes_phase")
    label: str = "mode_annealing"
    mode_schedule: dict[str, list[tuple[int, int]]] = field(
        default_factory=lambda: {
            0: [(2, 2)],
            1: [(2, 2), (2, 1)],
            2: [(2, 2), (2, 1), (3, 3)],
            3: [(2, 2), (2, 1), (3, 3), (3, 2)],
            4: [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4)],
        }
    )
    trigger_temperature: float = 0.9999
    n_particles: int = 500
    n_pool_processes: int = 8
    mutation_steps: int = 20
    ess_ratio: float = 0.8
    rng_seed: int = 1234
    bilby_seed: int = 88170235
    fixed_keys: Iterable[str] = field(
        default_factory=lambda: [
            "a_1",
            "a_2",
            "tilt_1",
            "tilt_2",
            "phi_12",
            "phi_jl",
            "psi",
            "ra",
            "dec",
        ]
    )
    additional_outputs: Dict[str, str] = field(
        default_factory=lambda: {
            "log_likelihood": "log_likelihood_vs_frequency.png",
            "metadata": "metadata_history.png",
            "intermediate_corner": "intermediate_corner_plots.png",
            "posterior_corner": "posterior_corner_plot.png",
        }
    )


class ModeAnnealingLikelihood(bilby.gw.GravitationalWaveTransient):
    """Extends the bilby likelihood with a helper for updating the max frequency."""

    def update_mode_array(self, mode_array) -> None:
        self.waveform_generator._cache = dict(
            parameters=None, waveform=None, model=None
        )
        self.waveform_generator.waveform_arguments["mode_array"] = mode_array


class AddModesStep(SMCStep):
    """
    Raises the maximum frequency once the system has effectively equilibrated at
    the current temperature. Resets the annealing state so the chain can adapt.
    """

    def __init__(
        self,
        mode_schedule: dict[str, list[tuple[int, int]]],
        trigger_temperature: float = 0.999,
    ):
        self._trigger_temperature = trigger_temperature
        self.mode_schedule = mode_schedule

    def apply(self, state, context, rng):
        temperature = float(context.shared.get("temperature", 0.0))
        if temperature < self._trigger_temperature:
            return state

        current_modes = context.shared.get("modes", [(2, 2)])
        modes_id = context.shared.get("modes_id", 0)
        # Get next entry in mode schedule
        new_id = modes_id + 1
        print("new id", new_id)
        next_modes = self.mode_schedule.get(new_id, [])
        if not next_modes:
            raise ValueError("No more modes to add in the schedule.")

        logger.info(
            "Mode step triggered at T=%.3f: %s -> %s",
            temperature,
            current_modes,
            next_modes,
        )

        context.shared["modes"] = next_modes
        context.shared["n_modes"] = len(next_modes)
        context.shared["modes_id"] = new_id
        context.shared["temperature"] = 0.0
        context.shared["temperature_delta"] = 0.0

        metadata = dict(state.metadata)
        metadata["modes"] = next_modes
        metadata["n_modes"] = context.shared["n_modes"]
        metadata["modes_id"] = new_id
        metadata["temperature"] = 0.0
        metadata.pop("temperature_delta", None)
        return state.with_updates(log_evidence=0.0, metadata=metadata)


def configure_example_logging(cfg: ModeAnnealingConfig) -> None:
    cfg.outdir.mkdir(parents=True, exist_ok=True)
    configure_logging(level="INFO")
    bilby.core.utils.setup_logger(outdir=str(cfg.outdir), label=cfg.label)
    bilby.core.utils.random.seed(cfg.bilby_seed)


def build_injection_parameters() -> Dict[str, float]:
    injection_parameters = dict(
        mass_1=36.0,
        mass_2=12.0,
        a_1=0.4,
        a_2=0.3,
        tilt_1=0.5,
        tilt_2=1.0,
        phi_12=1.7,
        phi_jl=0.3,
        luminosity_distance=500.0,
        theta_jn=0.4,
        psi=2.659,
        phase=1.3,
        geocent_time=1126259642.413,
        ra=1.375,
        dec=-1.2108,
    )
    return bilby.gw.conversion.generate_all_bbh_parameters(
        injection_parameters
    )


def build_waveform_generator(
    cfg: ModeAnnealingConfig,
) -> bilby.gw.WaveformGenerator:
    final_modes = cfg.mode_schedule[len(cfg.mode_schedule) - 1]
    logger.info("Final mode array for waveform generator: %s", final_modes)
    waveform_arguments = dict(
        waveform_approximant="IMRPhenomXPHM",
        reference_frequency=50.0,
        minimum_frequency=cfg.minimum_frequency,
        mode_array=final_modes,
    )
    return bilby.gw.WaveformGenerator(
        duration=cfg.duration,
        sampling_frequency=cfg.sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=waveform_arguments,
    )


def build_interferometers(
    cfg: ModeAnnealingConfig,
    injection_parameters: Dict[str, float],
    waveform_generator: bilby.gw.WaveformGenerator,
) -> bilby.gw.detector.InterferometerList:
    ifos = bilby.gw.detector.InterferometerList(["H1", "L1", "V1"])
    ifos.set_strain_data_from_power_spectral_densities(
        sampling_frequency=cfg.sampling_frequency,
        duration=cfg.duration,
        start_time=injection_parameters["geocent_time"] - 2,
    )
    ifos.inject_signal(
        waveform_generator=waveform_generator,
        parameters=injection_parameters,
    )
    return ifos


def build_priors(
    cfg: ModeAnnealingConfig,
    injection_parameters: Dict[str, float],
) -> bilby.gw.prior.BBHPriorDict:
    priors = bilby.gw.prior.BBHPriorDict()
    for key in cfg.fixed_keys:
        priors[key] = injection_parameters[key]
    priors = bilby.gw.prior.BBHPriorDict(priors)
    priors["chirp_mass"] = bilby.core.prior.Uniform(
        injection_parameters["chirp_mass"] - 5.0,
        injection_parameters["chirp_mass"] + 5.0,
        "chirp_mass",
    )
    priors["geocent_time"] = bilby.core.prior.Uniform(
        injection_parameters["geocent_time"] - 0.1,
        injection_parameters["geocent_time"] + 0.1,
        "geocent_time",
    )
    priors.validate_prior(cfg.duration, cfg.minimum_frequency)
    return priors


def get_initial_parameters(
    priors: bilby.core.prior.PriorDict,
) -> Dict[str, float]:
    parameters: Dict[str, float] = {}
    for key, prior in priors.items():
        if isinstance(prior, bilby.core.prior.DeltaFunction):
            parameters[key] = prior.sample()
    return parameters


def _logl_func(x, mode_array) -> float:
    from bilby.core.sampler.base_sampler import _sampling_convenience_dump

    priors = _sampling_convenience_dump.priors
    params = _sampling_convenience_dump.parameters.copy()
    updated_params = dict(zip(priors.non_fixed_keys, x))
    params.update(updated_params)
    _sampling_convenience_dump.likelihood.update_mode_array(mode_array)
    return _sampling_convenience_dump.likelihood.log_likelihood(params)


def scan_log_likelihood(
    likelihood: ModeAnnealingLikelihood,
    injection_parameters: Dict[str, float],
    mode_schedule: dict[str, list[tuple[int, int]]],
    output_path: Path,
) -> None:
    logl_values = []
    for modes in mode_schedule.values():
        likelihood.update_mode_array(modes)
        ll = likelihood.log_likelihood(injection_parameters)
        logl_values.append(ll)
    print("Log Likelihood values:", logl_values)
    fig, ax = plt.subplots()
    idx = np.arange(len(mode_schedule))
    ax.plot(idx, logl_values, marker="o")
    ax.set_xlabel("Mode Index")
    ax.set_ylabel("Log Likelihood")
    ax.set_title("Log Likelihood vs Mode Index")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def make_continue_annealing_predicate(
    final_n_modes: int,
    trigger_temperature: float,
):
    def predicate(state, context):
        temperature = float(state.metadata.get("temperature", 0.0))
        current_n_modes = state.metadata.get("n_modes", 1)
        if temperature < trigger_temperature:
            return True
        return current_n_modes < final_n_modes

    return predicate


def logging_str_fn(state, context) -> str:
    temperature = context.shared["temperature"]
    n_modes = context.shared["n_modes"]
    log_evidence = state.log_evidence
    return (
        f"it {context.t}:, "
        f"Temperature: {temperature:.4f}, "
        f"Number of Modes: {n_modes}, "
        f"Log Evidence: {log_evidence:.1f}"
    )


def build_posterior_functions(
    priors: bilby.core.prior.PriorDict,
    pool: mp.pool.Pool,
):
    def log_prior(particles, context):
        x = dict(zip(priors.non_fixed_keys, particles.T))
        return priors.ln_prob(x, axis=0)

    def log_likelihood(particles, context):
        modes = context.shared["modes"]
        logl = -np.inf * np.ones(len(particles), dtype=float)
        logp = log_prior(particles, context)
        if not np.any(np.isfinite(logp)):
            return logl
        mask = np.isfinite(logp, dtype=bool)
        x = np.asarray(particles[mask, :], dtype="float64")
        logl[mask] = np.fromiter(
            pool.starmap(_logl_func, zip(x, repeat(modes))),
            dtype=float,
        )
        return logl

    return log_prior, log_likelihood


def create_initial_state(
    priors: bilby.core.prior.PriorDict,
    mode_schedule: dict[str, list[tuple[int, int]]],
    n_particles: int,
) -> ParticleState:
    samples_dict = priors.sample(n_particles)
    samples = np.array([samples_dict[key] for key in priors.non_fixed_keys]).T
    log_weights = np.full((n_particles,), -np.log(n_particles))
    metadata = {
        "modes": mode_schedule[0],
        "n_modes": 1,
        "modes_id": 0,
        "temperature": 0.0,
    }
    return ParticleState(samples, log_weights=log_weights, metadata=metadata)


def run_smc(
    posterior: TemperedPosterior,
    likelihood: ModeAnnealingLikelihood,
    cfg: ModeAnnealingConfig,
    mode_schedule: dict[str, list[tuple[int, int]]],
    initial_state: ParticleState,
):
    target_n_modes = len(
        mode_schedule[len(mode_schedule) - 1]
    )  # Assumes modes are added one by one
    smc = SequentialMonteCarlo(
        steps=[
            AddModesStep(
                mode_schedule=mode_schedule,
                trigger_temperature=cfg.trigger_temperature,
            ),
            AdaptiveTemperedWeighting(
                posterior,
                target_ess_ratio=cfg.ess_ratio,
                max_temperature=1.0,
                min_delta=1e-8,
            ),
            SystematicResampling(always_resample=True),
            MiniPCNMutation(posterior, n_steps=cfg.mutation_steps),
            LoggingStep(logging_str_fn),
            ContinueWhile(
                make_continue_annealing_predicate(
                    final_n_modes=target_n_modes,
                    trigger_temperature=cfg.trigger_temperature,
                )
            ),
        ],
        auto_normalize=True,
    )
    rng = ArrayRNG("numpy", cfg.rng_seed)
    shared_context = {
        "temperature": 0.0,
        "temperature_delta": 0.0,
        "modes": mode_schedule[0],
        "n_modes": 1,
        "modes_id": 0,
    }
    return smc.run(initial_state, rng=rng, shared_context=shared_context)


def plot_metadata_history(result, output_path: Path) -> None:
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    result.plot_log_evidence(ax=axs[0])
    result.plot_metadata_history("n_modes", ax=axs[1])
    result.plot_metadata_history("temperature", ax=axs[2])
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_intermediate_corners(
    result,
    priors: bilby.core.prior.PriorDict,
    injection_parameters: Dict[str, float],
    output_path: Path,
) -> None:
    states_per_n_modes = {}
    for state in result.states:
        n_modes = state.metadata.get("n_modes", 1)
        states_per_n_modes.setdefault(n_modes, []).append(state)

    final_state_per_n_modes = {k: v[-1] for k, v in states_per_n_modes.items()}

    fig = None
    for i, state in enumerate(final_state_per_n_modes.values()):
        samples = state.particles
        fig = corner.corner(
            samples,
            labels=priors.non_fixed_keys,
            density=True,
            smooth=0.9,
            color=f"C{i}",
            truths=[
                injection_parameters[key] for key in priors.non_fixed_keys
            ],
            truth_color="C1",
            fig=fig,
        )
    if fig is not None:
        fig.savefig(output_path)
        plt.close(fig)


def plot_posterior_corner(
    result,
    priors: bilby.core.prior.PriorDict,
    injection_parameters: Dict[str, float],
    output_path: Path,
) -> None:
    posterior_samples = result.posterior_samples()
    corner_fig = corner.corner(
        posterior_samples,
        labels=priors.non_fixed_keys,
        density=True,
        smooth=0.9,
        color="C0",
        truths=[injection_parameters[key] for key in priors.non_fixed_keys],
        truth_color="C1",
    )
    corner_fig.savefig(output_path)
    plt.close(corner_fig)


def run_mode_annealing(cfg: ModeAnnealingConfig) -> None:
    configure_example_logging(cfg)

    # Save config to outdir
    import yaml

    with open(cfg.outdir / "config.yaml", "w") as f:
        yaml.dump(cfg.__dict__, f)

    injection_parameters = build_injection_parameters()
    waveform_generator = build_waveform_generator(cfg)
    ifos = build_interferometers(cfg, injection_parameters, waveform_generator)
    priors = build_priors(cfg, injection_parameters)
    likelihood = ModeAnnealingLikelihood(
        interferometers=ifos,
        waveform_generator=waveform_generator,
        distance_marginalization=True,
        time_marginalization=False,
        phase_marginalization=False,
        priors=priors,
    )
    logger.info("Starting annealing run with prior keys: %s", priors.keys())

    mode_schedule = cfg.mode_schedule

    scan_log_likelihood(
        likelihood,
        injection_parameters,
        mode_schedule,
        cfg.outdir / cfg.additional_outputs["log_likelihood"],
    )

    initial_state = create_initial_state(
        priors, mode_schedule, cfg.n_particles
    )

    likelihood.update_mode_array(mode_schedule[0])

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
        log_prior, log_likelihood = build_posterior_functions(priors, pool)
        posterior = TemperedPosterior(
            log_likelihood=log_likelihood,
            log_prior=log_prior,
        )
        result = run_smc(
            posterior,
            likelihood,
            cfg,
            mode_schedule,
            initial_state,
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
        result,
        priors,
        injection_parameters,
        cfg.outdir / cfg.additional_outputs["posterior_corner"],
    )
    logger.info("Finished run. Results stored in %s", cfg.outdir.resolve())


def main():
    run_mode_annealing(ModeAnnealingConfig())


if __name__ == "__main__":
    main()

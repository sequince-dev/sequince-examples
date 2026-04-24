import logging
import multiprocessing as mp
from dataclasses import dataclass, field
from itertools import repeat
from pathlib import Path
from typing import Dict, Iterable, Sequence

import bilby
import corner
import matplotlib.pyplot as plt
import numpy as np
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
from sequince.steps.control import StopIf
from sequince.steps.log import LoggingStep
from sequince.steps.nested import NestedSMCStep

logger = logging.getLogger(__name__)


@dataclass
class FrequencyAnnealingConfig:
    duration: float = 4.0
    sampling_frequency: float = 2048.0
    minimum_frequency: float = 20.0
    outdir: Path = Path("outdir_aligned_spin_auto_steps")
    label: str = "bilby_comparison"
    logging_level: int = logging.INFO
    frequency_schedule: Sequence[float] | str = (
        "auto"  # field(default_factory=lambda: [512, 1024, 2048])#"auto"
    )
    n_frequency_steps: int = 10
    trigger_temperature: float = 0.99
    n_particles: int = 1000
    n_pool_processes: int = 8
    mutation_steps: int = 100
    n_final_particles: int = 5000
    ess_ratio: float = 0.9
    rng_seed: int = 1234
    bilby_seed: int = 88170235
    zero_noise: bool = True
    waveform_approximant: str = "IMRPhenomD"
    aligned_spins: bool = True
    fixed_keys: Iterable[str] = field(default_factory=lambda: [])
    run_bilby_analysis: bool = True
    bilby_analysis_outdir: Path = Path("bilby_analysis")
    bilby_sampler_kwargs: Dict[str, object] = field(
        default_factory=lambda: {
            "sampler": "nessai",
            "nlive": 1000,
            "reset_flow": 8,
        }
    )
    additional_outputs: Dict[str, str] = field(
        default_factory=lambda: {
            "log_likelihood": "log_likelihood_vs_frequency.png",
            "metadata": "metadata_history.png",
            "intermediate_corner": "intermediate_corner_plots.png",
            "posterior_corner": "posterior_corner_plot.png",
        }
    )


class FrequencyAnnealingLikelihood(bilby.gw.GravitationalWaveTransient):
    """Extends the bilby likelihood with a helper for updating the max frequency."""

    def update_maximum_frequency(self, frequency: float) -> None:
        self.maximum_frequency = frequency
        for ifo in self.interferometers:
            ifo.maximum_frequency = frequency


def configure_example_logging(cfg: FrequencyAnnealingConfig) -> None:
    cfg.outdir.mkdir(parents=True, exist_ok=True)
    configure_logging(level=cfg.logging_level)
    bilby.core.utils.setup_logger(outdir=str(cfg.outdir), label=cfg.label)
    bilby.core.utils.random.seed(cfg.bilby_seed)


def build_injection_parameters(
    cfg: FrequencyAnnealingConfig,
) -> Dict[str, float]:
    if cfg.aligned_spins:
        injection_parameters = dict(
            mass_1=36.0,
            mass_2=29.0,
            chi_1=0.4,
            chi_2=0.3,
            luminosity_distance=1000.0,
            theta_jn=0.4,
            psi=2.659,
            phase=1.3,
            geocent_time=1126259642.413,
            ra=1.375,
            dec=-1.2108,
        )
    else:
        injection_parameters = dict(
            mass_1=36.0,
            mass_2=29.0,
            a_1=0.4,
            a_2=0.3,
            tilt_1=0.5,
            tilt_2=1.0,
            phi_12=1.7,
            phi_jl=0.3,
            luminosity_distance=1000.0,
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
    cfg: FrequencyAnnealingConfig,
) -> bilby.gw.WaveformGenerator:
    waveform_arguments = dict(
        waveform_approximant=cfg.waveform_approximant,
        reference_frequency=50.0,
        minimum_frequency=cfg.minimum_frequency,
    )
    return bilby.gw.WaveformGenerator(
        duration=cfg.duration,
        sampling_frequency=cfg.sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=waveform_arguments,
    )


def build_interferometers(
    cfg: FrequencyAnnealingConfig,
    injection_parameters: Dict[str, float],
    waveform_generator: bilby.gw.WaveformGenerator,
) -> bilby.gw.detector.InterferometerList:
    ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
    if cfg.zero_noise:
        ifos.set_strain_data_from_zero_noise(
            sampling_frequency=cfg.sampling_frequency,
            duration=cfg.duration,
            start_time=injection_parameters["geocent_time"] - 2,
        )
    else:
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
    cfg: FrequencyAnnealingConfig,
    injection_parameters: Dict[str, float],
) -> bilby.gw.prior.BBHPriorDict:
    priors = bilby.gw.prior.BBHPriorDict(aligned_spin=cfg.aligned_spins)
    for key in cfg.fixed_keys:
        priors[key] = injection_parameters[key]
    priors = bilby.gw.prior.BBHPriorDict(priors)
    priors["geocent_time"] = bilby.core.prior.Uniform(
        injection_parameters["geocent_time"] - 0.1,
        injection_parameters["geocent_time"] + 0.1,
        "geocent_time",
    )
    priors["chirp_mass"] = bilby.core.prior.Uniform(
        0.9 * injection_parameters["chirp_mass"],
        1.1 * injection_parameters["chirp_mass"],
        "chirp_mass",
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


def _logl_func(x, max_freq):
    from bilby.core.sampler.base_sampler import _sampling_convenience_dump

    priors = _sampling_convenience_dump.priors
    params = _sampling_convenience_dump.parameters.copy()
    updated_params = dict(zip(priors.non_fixed_keys, x))
    params.update(updated_params)
    _sampling_convenience_dump.likelihood.update_maximum_frequency(max_freq)
    return _sampling_convenience_dump.likelihood.log_likelihood(params)


def determine_frequency_schedule(
    likelihood: FrequencyAnnealingLikelihood,
    injection_parameters: Dict[str, float],
    start_frequency: float,
    end_frequency: float,
    n_grid: int,
    n_steps: int = 10,
) -> np.ndarray:
    frequencies = np.linspace(start_frequency, end_frequency, n_grid)
    logl_values = []
    for f in frequencies:
        likelihood.update_maximum_frequency(float(f))
        logl_values.append(likelihood.log_likelihood(injection_parameters))
    logl_values = np.array(logl_values)
    deltas = np.abs(np.diff(logl_values))
    # Select n_steps frequencies that are evenly spaced in delta log-likelihood
    cumulative_deltas = np.insert(np.cumsum(deltas), 0, 0.0)
    total_delta = cumulative_deltas[-1]
    target_deltas = np.linspace(0, total_delta, n_steps)
    frequency_schedule = []
    for td in target_deltas:
        idx = np.searchsorted(cumulative_deltas, td)
        frequency_schedule.append(frequencies[idx])
    return np.array(sorted(set(frequency_schedule)), dtype=float)


def scan_log_likelihood(
    likelihood: FrequencyAnnealingLikelihood,
    injection_parameters: Dict[str, float],
    frequency_schedule: np.ndarray,
    output_path: Path,
) -> None:
    logl_values = []
    for f in frequency_schedule:
        likelihood.update_maximum_frequency(float(f))
        logl_values.append(likelihood.log_likelihood(injection_parameters))
    fig, ax = plt.subplots()
    ax.plot(frequency_schedule, logl_values, marker="o")
    ax.set_xlabel("Maximum Frequency (Hz)")
    ax.set_ylabel("Log Likelihood")
    ax.set_title("Log Likelihood vs Maximum Frequency")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def make_continue_annealing_predicate(
    final_maximum_frequency: float,
    trigger_temperature: float,
):
    def predicate(state, context):
        temperature = float(state.metadata.get("temperature", 0.0))
        current_frequency = state.metadata.get(
            "current_maximum_frequency", 0.0
        )
        if temperature < trigger_temperature:
            return True
        return current_frequency < final_maximum_frequency

    return predicate


def logging_str_fn(state, context) -> str:
    temperature = context.metadata["temperature"]
    maximum_frequency = context.metadata.get("current_maximum_frequency", 0.0)
    log_evidence = state.log_evidence
    return (
        f"it {context.t}:, "
        f"Temperature: {temperature:.4f}, "
        f"Maximum Frequency: {maximum_frequency:.1f}, "
        f"Log Evidence: {log_evidence:.1f}"
    )


def build_posterior_functions(
    priors: bilby.core.prior.PriorDict,
    likelihood: FrequencyAnnealingLikelihood,
    pool: mp.Pool,
):
    def log_prior(particles, context):
        x = dict(zip(priors.non_fixed_keys, particles.T))
        return priors.ln_prob(x, axis=0)

    def log_likelihood(particles, context):
        max_freq = context.metadata["current_maximum_frequency"]
        likelihood.update_maximum_frequency(max_freq)
        logl = -np.inf * np.ones(len(particles), dtype=float)
        logp = log_prior(particles, context)
        # If the maximum frequency is at the minimum, we can skip the likelihood
        # evaluation and just return zeros
        if max_freq == context.shared["minimum_frequency"]:
            return np.zeros_like(logp)
        if not np.any(np.isfinite(logp)):
            return logl
        mask = np.isfinite(logp, dtype=bool)
        x = np.asarray(particles[mask, :], dtype="float64")
        logl[mask] = np.fromiter(
            pool.starmap(_logl_func, zip(x, repeat(max_freq))),
            dtype=float,
        )
        return logl

    return log_prior, log_likelihood


def create_initial_state(
    priors: bilby.core.prior.PriorDict,
    frequency_schedule: np.ndarray,
    n_particles: int,
) -> ParticleState:
    samples_dict = priors.sample(n_particles)
    samples = np.array([samples_dict[key] for key in priors.non_fixed_keys]).T
    log_weights = np.full((n_particles,), -np.log(n_particles))
    metadata = {
        "maximum_frequency": float(frequency_schedule[0]),
        "temperature": 0.0,
    }
    return ParticleState(samples, log_weights=log_weights, metadata=metadata)


def run_bilby_analysis(
    cfg: FrequencyAnnealingConfig,
    priors,
    likelihood,
    injection_parameters,
    **kwargs,
):
    """Run a standard bilby analysis for each frequency in the schedule for comparison."""
    results = {}

    if cfg.frequency_schedule == "auto":
        frequency_schedule = determine_frequency_schedule(
            likelihood,
            injection_parameters,
            start_frequency=20.0,
            end_frequency=cfg.sampling_frequency / 2,
            n_grid=500,
            n_steps=cfg.n_frequency_steps,
        )
    else:
        frequency_schedule = np.array(
            sorted(set(cfg.frequency_schedule)), dtype=float
        )

    for freq in frequency_schedule:
        logger.info(
            "Running bilby analysis with maximum frequency %.1f Hz", freq
        )
        sampler_config = cfg.bilby_sampler_kwargs.copy()
        sampler_config.update(kwargs)

        likelihood.update_maximum_frequency(float(freq))
        result = bilby.run_sampler(
            likelihood=likelihood,
            priors=priors,
            outdir=cfg.bilby_analysis_outdir,
            label=f"bilby_{int(freq)}Hz",
            npool=cfg.n_pool_processes,
            **sampler_config,
        )
        logger.info(
            "Completed bilby analysis for max frequency %.1f Hz. Results stored in %s",
            freq,
            cfg.bilby_analysis_outdir.resolve(),
        )
        results[freq] = result
    return results


def plot_metadata_history(result, output_path: Path) -> None:
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    result.plot_log_evidence(ax=axs[0])
    result.plot_metadata_history("maximum_frequency", ax=axs[1])
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
    states_per_maximum_frequency = {}
    for state in result.states:
        max_freq = state.metadata.get("current_maximum_frequency", 0.0)
        states_per_maximum_frequency.setdefault(max_freq, []).append(state)

    final_state_per_maximum_frequency = {
        k: v[-1] for k, v in states_per_maximum_frequency.items()
    }
    colors = plt.cm.magma(
        np.linspace(0, 1, len(final_state_per_maximum_frequency))
    )
    fig = None
    for i, state in enumerate(final_state_per_maximum_frequency.values()):
        samples = state.particles
        fig = corner.corner(
            samples,
            labels=priors.non_fixed_keys,
            density=True,
            smooth=0.9,
            color=colors[i],
            truths=[
                injection_parameters[key] for key in priors.non_fixed_keys
            ],
            truth_color="grey",
            fig=fig,
        )
    if fig is not None:
        fig.savefig(output_path)
        plt.close(fig)


def plot_posterior_corner(
    state,
    priors: bilby.core.prior.PriorDict,
    injection_parameters: Dict[str, float],
    output_path: Path,
) -> None:
    posterior_samples = state.particles
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


def plot_comparison_to_reference(
    reference_results: Dict[float, bilby.core.result.Result],
    result,
    priors: bilby.core.prior.PriorDict,
    injection_parameters: Dict[str, float],
    output_path: Path,
) -> None:
    """Produce a comparison per frequency bin between the SMC posterior and the bilby sampler results."""
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Plotting comparison to reference results for frequencies: %s",
        sorted(reference_results.keys()),
    )
    states_per_maximum_frequency = {}
    for state in result.states:
        max_freq = state.metadata.get("current_maximum_frequency", 0.0)
        states_per_maximum_frequency.setdefault(max_freq, []).append(state)

    final_state_per_maximum_frequency = {
        k: v[-1] for k, v in states_per_maximum_frequency.items()
    }
    logger.info(
        "Final frequencies in SMC result: %s",
        sorted(final_state_per_maximum_frequency.keys()),
    )
    # Only compare frequencies for which we have reference results
    for freq in final_state_per_maximum_frequency.keys():
        if freq not in reference_results:
            logger.warning(
                "No reference result for frequency %.1f Hz. Skipping comparison plot.",
                freq,
            )
            continue
        state = final_state_per_maximum_frequency[freq]
        samples = state.particles
        reference_samples = (
            reference_results[freq].posterior[priors.non_fixed_keys].values
        )

        corner_kwargs = dict(
            labels=priors.non_fixed_keys,
            density=True,
            smooth=0.9,
            truths=[
                injection_parameters[key] for key in priors.non_fixed_keys
            ],
            truth_color="k",
            color="C0",
            hist_kwargs=dict(color="C0", histtype="step", density=True),
            no_fill_contours=True,
        )

        fig = corner.corner(
            samples,
            fig=None,
            **corner_kwargs,
        )
        corner_kwargs["color"] = "C1"
        corner_kwargs["hist_kwargs"]["color"] = "C1"
        corner.corner(
            reference_samples,
            fig=fig,
            **corner_kwargs,
        )
        fig.suptitle(f"Comparison at max frequency {freq:.1f} Hz")
        fig.savefig(output_path / f"comparison_{int(freq)}Hz.png")
        # Add legend manually since corner doesn't support legends well
        handles = [
            plt.Line2D([], [], color="C0", label="SMC Posterior"),
            plt.Line2D([], [], color="C1", label="Reference Posterior"),
        ]
        fig.legend(handles=handles, loc="upper right")
        plt.close(fig)


class IncreaseFrequency(SMCStep):
    def __init__(self, frequency_schedule: np.ndarray):
        self.frequency_schedule = frequency_schedule

    def apply(self, state, context, rng):
        current_frequency = context.metadata["current_maximum_frequency"]
        target_frequency = context.shared["maximum_frequency"]
        frequency_schedule = self.frequency_schedule
        schedule_idx = np.searchsorted(frequency_schedule, current_frequency)
        next_idx = min(schedule_idx + 1, len(frequency_schedule) - 1)
        next_frequency = min(frequency_schedule[next_idx], target_frequency)
        context.metadata["current_maximum_frequency"] = next_frequency
        state_metadata = dict(state.metadata)
        state_metadata["current_maximum_frequency"] = next_frequency
        state = state.with_updates(metadata=state_metadata)
        logger.info(
            "Transitioning from max frequency %.1f Hz to %.1f Hz",
            current_frequency,
            next_frequency,
        )
        return state


def frequency_predicate(state, context):
    current_frequency = context.metadata["current_maximum_frequency"]
    return current_frequency == context.shared["frequency_schedule"][-1]


def run_smc(
    posterior: TemperedPosterior,
    cfg: FrequencyAnnealingConfig,
    frequency_schedule: np.ndarray,
    initial_state: ParticleState,
):
    initial_metadata = {
        "current_maximum_frequency": 20.0,
    }

    shared_context = {
        "maximum_frequency": float(frequency_schedule[-1]),
        "minimum_frequency": float(20.0),
        "frequency_schedule": frequency_schedule,
    }

    inner_runner = SequentialMonteCarlo(
        steps=[
            AdaptiveTemperedWeighting(
                posterior,
                target_ess_ratio=cfg.ess_ratio,
                max_temperature=1.0,
                min_delta=1e-8,
            ),
            SystematicResampling(always_resample=True),
            MiniPCNMutation(posterior, n_steps=cfg.mutation_steps),
            LoggingStep(logging_str_fn),
        ],
    )

    smc = SequentialMonteCarlo(
        steps=[
            IncreaseFrequency(frequency_schedule=frequency_schedule),
            NestedSMCStep(
                inner_runner, posterior=posterior, record_outputs=True
            ),
            StopIf(frequency_predicate),
        ],
    )
    rng = ArrayRNG("numpy", cfg.rng_seed)
    shared_context = {
        "maximum_frequency": float(frequency_schedule[-1]),
        "minimum_frequency": float(20.0),
        "frequency_schedule": frequency_schedule,
    }
    metadata = initial_metadata
    return smc.run(
        initial_state,
        rng=rng,
        shared_context=shared_context,
        metadata=metadata,
    )

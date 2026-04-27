import numpy as np
import matplotlib.pyplot as plt
from nessai.model import Model
from nessai.flowsampler import FlowSampler
from nessai.plot import corner_plot
from nessai.utils.logging import configure_logger
import pathlib
from hypertriangle import hypertriangulate, hypercubify


def gauss(x, A, mu, sigma):
    """Single Gaussian pulse model."""
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

class PulseModel(Model):
    def __init__(self, n_pulses, pulse_bounds, xdata, ydata, noise_sigma):
        # Names of parameters to sample
        names = []
        for i in range(n_pulses):
            names += [f"A_{i}", f"mu_{i}", f"sigma_{i}"]
        self.names = names

        bounds = {}
        for i in range(n_pulses):
            bounds[f"A_{i}"] = pulse_bounds["amplitude_bounds"]
            bounds[f"mu_{i}"] = pulse_bounds["mean_bounds"]
            bounds[f"sigma_{i}"] = pulse_bounds["scale_bounds"]

        self.bounds = bounds

        self.n_pulses = n_pulses
        self.xdata = xdata
        self.ydata = ydata
        self.noise_sigma = noise_sigma

    def log_prior(self, x):
        log_p = np.log(self.in_bounds(x), dtype="float")
        for n in self.names:
            log_p -= np.log(self.bounds[n][1] - self.bounds[n][0])
        return log_p

    def log_likelihood(self, x):
        parameters_in = np.zeros((3, self.n_pulses, x.size))
        for i in range(self.n_pulses):
            parameters_in[0, i] = x[f"A_{i}"]
            parameters_in[1, i] = x[f"mu_{i}"]
            parameters_in[2, i] = x[f"sigma_{i}"]

        # apply hypertriangle map
        parameters_in[1] = hypertriangulate(parameters_in[1].T, bounds=self.bounds["mu_0"]).T
        templates = gauss(self.xdata[:, None, None], *parameters_in).sum(1)
        log_norm = -0.5 * self.xdata.size * np.log(2 * np.pi * self.noise_sigma**2)
        logl = (
            -0.5 * ((self.ydata[:, None] - templates) ** 2).sum(0) / (self.noise_sigma**2)
            + log_norm
        )
        return logl
    

rng = np.random.default_rng(1730)

bounds = np.array(
    [
        [0.1, 1.5],
        [-4, 4],
        [0.1, 1],
    ]
).T

n_pulses = 2

outdir = pathlib.Path("nessai") / f"{n_pulses}_pulses"
outdir.mkdir(parents=True, exist_ok=True)

configure_logger(output=outdir)

A_true, mu_true, sigma_true = rng.uniform(
    bounds[0], bounds[1], (n_pulses, 3)
).T

print(f"True parameters: A={A_true}, mu={mu_true}, sigma={sigma_true}")

xdata = np.linspace(-5, 5, 101)
signals = gauss(xdata[:, None], A_true, mu_true, sigma_true)
signal_tot = signals.sum(1)
noise_sigma = 1e-3
noise = noise_sigma * rng.normal(size=xdata.shape)
ydata = signal_tot + noise

plt.plot(xdata, ydata)
plt.plot(xdata, signal_tot)
plt.plot(xdata, signals, alpha=0.5, c="k")
plt.savefig(outdir / "data.png")
plt.close()


model = PulseModel(
    n_pulses=n_pulses,
    pulse_bounds={
        "amplitude_bounds": bounds[:,0],
        "mean_bounds": bounds[:,1],
        "scale_bounds": bounds[:,2],
    },
    xdata=xdata,
    ydata=ydata,
    noise_sigma=noise_sigma,
)

fs = FlowSampler(model, output=outdir, rng=rng, nlive=10000, resume=True)
fs.run()

print("Log evidence:", fs.log_evidence)
print("Log evidence error:", fs.log_evidence_error)

truths_stack = np.vstack((A_true, mu_true, sigma_true)).T
truths_stack = truths_stack[np.argsort(truths_stack[:, 1])]

# apply hypertriangle map to the means for the corner plot
tmp = np.zeros((n_pulses, fs.posterior_samples.size))
for i in range(n_pulses):
    tmp[i] = fs.posterior_samples[f"mu_{i}"]

tmp = hypertriangulate(tmp.T, bounds=bounds[:,1]).T

for i in range(n_pulses):
    fs.posterior_samples[f"mu_{i}"] = tmp[i]

corner_plot(
    fs.posterior_samples,
    exclude=["logL", "logP", "it"],
    truths=truths_stack.flatten(),
    filename=outdir / "corner.png",
)
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm


class Environment:
    def __init__(self, number_of_customers) -> None:

        self.number_of_customers = number_of_customers

    def create_experiment(self, plot=False):
        distributions_clips = np.random.randn(self.number_of_customers, 2)
        distributions_clips -= np.min(distributions_clips)
        distributions_clips *= 10
        distributions_clips.sort(axis=1)
        means = (
            np.random.random(self.number_of_customers)
            * (distributions_clips[:, 1] - distributions_clips[:, 0])
            + distributions_clips[:, 0]
        )

        stds = np.random.random(self.number_of_customers) * 3 + 0.8

        periods = (distributions_clips - means[:, np.newaxis]) / stds[:, np.newaxis]
        a, b = periods[:, 0], periods[:, 1]

        expectations = np.zeros(self.number_of_customers)
        medians = np.zeros(self.number_of_customers)
        samples = np.zeros(self.number_of_customers)
        dists = []
        for i in range(len(a)):
            dist = stats.truncnorm(a[i], b[i], loc=means[i], scale=stds[i])
            expectations[i] = dist.expect()
            medians[i] = dist.median()
            samples[i] = dist.rvs()
            dists.append(dist)

        if plot:
            self.plot_distros(a, b, means, stds, expectations, medians, samples)

        return expectations, medians, samples, dists

    def plot_distros(self, a, b, means, stds, expectations, medians, samples):
        color = cm.rainbow(np.linspace(0, 1, len(a)))

        x = np.linspace(
            stats.truncnorm.ppf(0.01, a, b, loc=means, scale=stds),
            stats.truncnorm.ppf(0.99, a, b, loc=means, scale=stds),
            100,
        )
        fig, ax = plt.subplots(1, 1)
        for i in range(len(a)):

            ax.plot(
                x[:, i],
                stats.truncnorm.pdf(x[:, i], a[i], b[i], loc=means[i], scale=stds[i]),
                lw=5,
                alpha=0.6,
                label="truncnorm pdf",
                color=color[i],
            )

            ax.axvline(expectations[i], 0, 1, color=color[i], linestyle="--")
            ax.axvline(medians[i], 0, 1, color=color[i], linestyle="-")
            ax.axvline(samples[i], 0, 1, color=color[i], linestyle="-.")
        plt.show()


if __name__ == "__main__":
    env = Environment(10)
    print(env.create_experiment())

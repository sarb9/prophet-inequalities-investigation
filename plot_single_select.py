import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == "__main__":
    results = np.load("./results/results_single.npy")

    results_cum = np.cumsum(results, axis=0)

    sns.set(style="whitegrid")

    lambda_ratios = results_cum[:, 0] / results_cum[:, 2]
    eta_ratios = results_cum[:, 1] / results_cum[:, 2]
    plt.plot(lambda_ratios, "r", label="Lambda: $\lambda = 1/2 \mathbb{E}[\max_i X_i]$")
    plt.plot(eta_ratios, "b", label="Eta: $Pr[\max_i X_i \geq \eta$] > 1/2")
    plt.legend()
    plt.title("Performance of the Prophet Inequalities")
    plt.xlabel("Number of Experiments")
    plt.ylabel("Competitive Ratio")
    # save the figure
    plt.savefig("./plots/single_select.png")
    plt.show()

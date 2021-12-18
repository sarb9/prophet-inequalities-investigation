import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == "__main__":
    comp_ratio = np.load("./results/results_two_ratio.npy")
    import pdb

    pdb.set_trace()

    sns.set(style="whitegrid")

    plt.plot(
        comp_ratio, "b", label="w: $\mathbb{E}[\sum_{i=1}^n I\{X_i \geq w\}] = k$",
    )
    plt.legend()
    plt.title("Performance of the Prophet Inequalities")
    plt.xlabel("Number of Experiments")
    plt.ylabel("Competitive Ratio")
    # save the figure
    plt.savefig("./plots/multiple_select.png")
    plt.show()

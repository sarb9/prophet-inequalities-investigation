from environment import Environment
import numpy as np

number_of_customers = 100
number_of_experiments = 1000

if __name__ == "__main__":
    env = Environment(number_of_customers)
    results = np.zeros((number_of_experiments, 3))
    for i in range(number_of_experiments):
        print(i, "start")
        expectations, medians, samples, _ = env.create_experiment()

        max_idx = np.argmax(expectations)

        lambda_ = expectations[max_idx] / 2
        eta = medians[max_idx]

        lambda_selected = None
        eta_selected = None
        for price in samples:
            if price > lambda_ and lambda_selected is None:
                lambda_selected = price
            if price > eta and eta_selected is None:
                eta_selected = price

            if eta_selected is not None and lambda_selected is not None:
                break

        if eta_selected is None:
            eta_selected = samples[-1]

        results[i] = [lambda_selected, eta_selected, expectations[max_idx]]
        print("lambda selected:", lambda_selected, "eta selected:", eta_selected)
        print("lambda:", lambda_, "eta:", eta, "expectation:", expectations[max_idx])

    np.save("results/results_single.npy", results)
    lambda_selected_mean, eta_selected_mean, expected_max = np.mean(results, axis=0)
    print(lambda_selected_mean, eta_selected_mean, expected_max)

from environment import Environment
import numpy as np

number_of_customers = 500
number_of_experiments = 1000
k = 20


def expected(threshold, dists):
    ans = 0
    for dist in dists:
        ans += dist.sf(threshold)

    return ans


def binary_search(start, end, target, dists):
    t = start + (end - start) / 2
    expected_number_of_bigger_than_t = expected(t, dists)
    if np.abs(expected_number_of_bigger_than_t - target) < 0.01:
        return t
    if expected_number_of_bigger_than_t > target:
        return binary_search(t, end, target, dists)
    return binary_search(start, t, target, dists)


if __name__ == "__main__":
    env = Environment(number_of_customers)

    selecteds = np.zeros((number_of_experiments, k))
    bests = np.zeros((number_of_experiments, k))
    for i in range(number_of_experiments):
        print(i)

        expectations, medians, samples, dists = env.create_experiment()

        delta = np.sqrt(2 * k * np.log(k))
        target = k - delta

        T = binary_search(0, np.max(expectations) * 2, target, dists)
        print(f"T: {T}, target: {target}, max: {np.max(expectations)}")

        selected_idx = 0
        for price in samples:
            if price > T:
                selecteds[i, selected_idx] = price
                selected_idx += 1
                if selected_idx == k:
                    break
        if selected_idx != k:
            selecteds[i, selected_idx:] = samples[-(k - selected_idx) :]

        bests[i] = np.partition(samples, -k)[-k:]

    sum_bests = np.sum(bests, axis=1)
    sum_selecteds = np.sum(selecteds, axis=1)
    best_cum = np.cumsum(sum_bests)
    selected_cum = np.cumsum(sum_selecteds)
    ratio = selected_cum / best_cum

    np.save("results/results_multiple_ratio.npy", ratio)

    np.save("results/results_multiple_bests.npy", bests)
    np.save("results/results_multiple_selecteds.npy", selecteds)

from environment import Environment
import numpy as np

number_of_customers = 10

if __name__ == "__main__":
    env = Environment(number_of_customers)
    env.create_experiment(plot=True)

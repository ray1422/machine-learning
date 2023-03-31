import random
import time

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

matplotlib.use("TKAgg")
MISLABELLED = False
M = 0.3
B = 2.1
random.seed(48763)
np.random.seed(48763)


def gen_samples(n_pos, n_neg, n_mislabelled_per_cls):
    samples = []
    # positive samples
    for i in range(n_pos):
        x = random.random()
        y = M * x + B + random.random() + 0.1
        samples.append([x, y, 1])

    # negative samples
    for i in range(n_neg):
        x = random.random()
        y = M * x + B - random.random() - 0.1
        samples.append([x, y, -1])

    for i in range(n_mislabelled_per_cls):
        samples[i][-1] *= -1
        samples[-i - 1][-1] *= -1

    samples = np.asarray(samples)
    np.random.shuffle(samples)
    return samples


def timer_and_result(*arg_names):
    def decorator(func):
        def wrapper(*args, **kwargs):
            t1 = time.time()
            result = func(*args, **kwargs)
            t2 = time.time()
            print(f'{func.__name__:20s} {(t2 - t1):.4f}s', end="")
            for i, arg_name in enumerate(arg_names):
                print(f", {arg_name}: {result[i]}", end="")
            print("")
            return result

        return wrapper

    return decorator


@timer_and_result("epoch", "step")
def pla(samples) -> (int, int):
    weights = np.random.random((2,))
    bias = np.random.uniform()
    epoch = 0
    step = 0
    for epoch in range(100000):
        np.random.shuffle(samples)
        all_correct = True
        for step, sample in enumerate(samples):
            x = sample[0:2]
            y = sample[2]
            y_pred = 1 if np.dot(weights.T, x) + bias > 0 else -1
            if y_pred != y:
                all_correct = False
                weights += np.float32(x) * np.float32(y)
                bias += np.float32(y)
        if all_correct:
            break
    return epoch, step

@timer_and_result("epoch", "step", "correct_cnt")
def pocket_pla(samples) -> (int, int):
    weights = np.random.random((2,))
    bias = np.random.uniform()
    epoch = 0
    step = 0
    correct_cnt = 0
    all_correct = False
    for epoch in range(30):
        for step, sample in enumerate(samples):
            x = sample[0:2]
            y = sample[2]
            y_pred = 1 if np.dot(weights, x) + bias > 0 else -1
            weights_bak = weights
            bias_bak = bias
            correct_cnt_bak = correct_cnt
            if y_pred != y:
                weights += np.float32(x) * np.float32(y)
                bias += np.float32(y)
                pred_all = np.matmul(np.asarray([weights]), samples[:, 0:2].T)[0] + bias
                correct_cnt = np.sum((pred_all > 0) == (samples[:, 2] > 0))
                if correct_cnt < correct_cnt_bak:
                    weights = weights_bak
                    bias = bias_bak
                if correct_cnt == len(samples):
                    all_correct = True
                    break
        if all_correct:
            break
        np.random.shuffle(samples)
    return epoch, step, correct_cnt


def problem_2():
    print("============= problem 2 =============")
    samples = gen_samples(15, 15, 0)
    pla(samples)


def problem_3():
    print("============= problem 3 =============")
    samples = gen_samples(1000, 1000, 0)
    pla(samples)
    pocket_pla(samples)

def problem_4():
    print("============= problem 4 =============")
    samples = gen_samples(1000, 1000, 50)
    # plt.scatter(x=np.asarray(samples)[:, 0], y=np.asarray(samples)[:, 1], c=np.asarray(samples)[:, 2])
    # plt.show()
    pocket_pla(samples)



problem_2()
print()
problem_3()
print()
problem_4()

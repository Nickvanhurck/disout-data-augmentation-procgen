import os
import csv
import numpy as np
import matplotlib.pyplot as plt

log_path = "logs/procgen_env/coinrun/"

baseline_path = os.path.join(log_path, "baseline")
data_aug_path = os.path.join(log_path, "data_aug")
disout_path = os.path.join(log_path, "disout")
disout_data_aug_path = os.path.join(log_path, "disout_data_aug")

runs = 8
labels = ["baseline", "data augmentation", "disout", "disout + data augmentation"]
rewards = [[0 for _ in range(1526)] for _ in range(4)]
rewards_time = [[0 for _ in range(1526)] for _ in range(4)]

for index, path in enumerate([baseline_path, data_aug_path, disout_path, disout_data_aug_path]):
    for directory in os.listdir(path):
        # print(directory)
        dir_path = os.path.join(path, directory)
        # print(dir_path)
        for file in os.listdir(dir_path):
            if file.endswith(".csv"):
                file_path = os.path.join(dir_path, file)
                with open(file_path, newline='') as csvfile:
                    logs = csv.reader(csvfile, delimiter=',', quotechar='|')
                    for i, log in enumerate(logs):
                        if i != 0:
                            rewards_time[index][i - 1] += float(log[0])
                            rewards[index][i - 1] += float(log[4])

# print()
# print("Results: ")

new_rewards = [[] for _ in range(4)]
new_rewards_time = [[] for _ in range(4)]

for i in range(4):
    for x in rewards[i]:
        new_rewards[i].append(x / runs)

    for x in rewards_time[i]:
        new_rewards_time[i].append(x / runs)

    plt.plot(new_rewards_time[i], np.cumsum(new_rewards[i]), alpha=0.9, label=labels[i])
    # plt.plot(new_rewards_time[i], new_rewards[i], alpha=0.9, label=labels[i])

plt.xlabel('episodes')
plt.ylabel('reward')
plt.legend()

plt.savefig("plots/cum_rewards.pdf")

plt.show()

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

for env in ["bigfish", "coinrun", "heist", "starpilot"]:

    runs = 8
    labels = ["baseline", "crop data augmentation", "disout", "disout + crop data augmentation"]

    # training
    log_train_path = "logs/training/" + env + "/"

    training_baseline_path = os.path.join(log_train_path, "baseline")
    training_data_aug_path = os.path.join(log_train_path, "data_aug")
    training_disout_path = os.path.join(log_train_path, "disout")
    training_disout_data_aug_path = os.path.join(log_train_path, "disout_data_aug")

    training_rewards = [[0 for _ in range(1526)] for _ in range(4)]
    training_rewards_time = [[0 for _ in range(1526)] for _ in range(4)]

    trainset = [training_baseline_path, training_data_aug_path, training_disout_path, training_disout_data_aug_path]

    for index, path in enumerate(trainset):
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
                                training_rewards_time[index][i - 1] += float(log[0])
                                training_rewards[index][i - 1] += float(log[4])

    training_new_rewards = [[] for _ in range(len(trainset))]
    training_new_rewards_time = [[] for _ in range(len(trainset))]

    for i in range(len(trainset)):
        for x in training_rewards[i]:
            training_new_rewards[i].append(x / runs)

        for x in training_rewards_time[i]:
            training_new_rewards_time[i].append(x / runs)

        # plt.plot(training_new_rewards_time[i], np.cumsum(training_new_rewards[i]), alpha=0.9, label=labels[i])
        plt.plot(training_new_rewards_time[i], training_new_rewards[i], alpha=0.9, label=labels[i])

    plt.xlabel('episodes')
    plt.ylabel('reward')
    plt.legend()

    plt.savefig("plots/" + env + "/training_rewards.pdf")
    # plt.savefig("plots/" + env + "/training_cum_rewards.pdf")

    plt.close()
    # plt.show()

    # testing
    log_test_path = "logs/testing/" + env + "/logs/training/" + env

    testing_baseline_path = os.path.join(log_test_path, "baseline")
    testing_data_aug_path = os.path.join(log_test_path, "data_aug")
    testing_disout_path = os.path.join(log_test_path, "disout")
    testing_disout_data_aug_path = os.path.join(log_test_path, "disout_data_aug")

    testing_rewards = [[0 for _ in range(1526)] for _ in range(4)]
    testing_rewards_time = [[0 for _ in range(1526)] for _ in range(4)]

    # for index, path in enumerate([testing_baseline_path, testing_data_aug_path, testing_disout_path, testing_disout_data_aug_path]):
    testset = [testing_baseline_path, testing_data_aug_path]
    for index, path in enumerate(testset):
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
                                testing_rewards_time[index][i - 1] += float(log[0])
                                testing_rewards[index][i - 1] += float(log[4])

    testing_new_rewards = [[] for _ in range(len(testset))]
    testing_new_rewards_time = [[] for _ in range(len(testset))]

    for i in range(len(testset)):
        for x in testing_rewards[i]:
            testing_new_rewards[i].append(x / runs)

        for x in testing_rewards_time[i]:
            testing_new_rewards_time[i].append(x / runs)

        # plt.plot(testing_new_rewards_time[i], np.cumsum(testing_new_rewards[i]), alpha=0.9, label=labels[i])
        plt.plot(testing_new_rewards_time[i], testing_new_rewards[i], alpha=0.9, label=labels[i])

    plt.xlabel('episodes')
    plt.ylabel('reward')
    plt.legend()

    plt.savefig("plots/" + env + "/testing_rewards.pdf")
    # plt.savefig("plots/" + env + "/testing_cum_rewards.pdf")

    plt.close()
    # plt.show()

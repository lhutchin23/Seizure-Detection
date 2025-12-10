import math
import random

import numpy as np


def Normalize(data):
    """
    Takes in a 2D numpy array and returns Z-score normalization using a global mean and std dev.
    """
    total_sum = 0
    num_elements = 0
    for row in data:
        total_sum += sum(row)
        num_elements += len(row)
    mean = total_sum / num_elements

    variance = 0
    for row in data:
        for x in row:
            variance += (x - mean) ** 2
    variance /= num_elements
    std_dev = math.sqrt(variance)

    if std_dev == 0:
        std_dev = 1

    normalized_data = [(row - mean) / std_dev for row in data]

    return normalized_data, mean, std_dev


def preprocess():
    np.random.seed(42)
    random.seed(42)

    SeizureDataPath = "../DATA/seizure.txt"
    NonSeizureDataPath = "../DATA/nonseizure.txt"
    SeizureData = []
    NonSeizureData = []

    with open(SeizureDataPath, "r") as f:
        for line in f:
            lst = list(map(int, line.strip().split(",")))
            SeizureData.append(lst)

    with open(NonSeizureDataPath, "r") as f:
        for line in f:
            lst = list(map(int, line.strip().split(",")))
            NonSeizureData.append(lst)

    print(
        "Data loaded: {} seizure samples, {} non-seizure samples".format(
            len(SeizureData), len(NonSeizureData)
        )
    )

    #Each subject has 23 chunks of 1-second EEG data
    chunks_per_subject = 23
    num_seizure_subjects = len(SeizureData) // chunks_per_subject  # 100 subjects
    num_nonseizure_subjects = len(NonSeizureData) // chunks_per_subject  # 400 subjects

    #Create subject IDs for splitting
    seizure_subjects = np.arange(num_seizure_subjects)
    nonseizure_subjects = np.arange(num_nonseizure_subjects)

    np.random.shuffle(seizure_subjects)
    np.random.shuffle(nonseizure_subjects)

    # 80:10:10 split on subjects
    sz_train_split = int(0.8 * num_seizure_subjects)
    sz_val_split = int(0.9 * num_seizure_subjects)

    nsz_train_split = int(0.8 * num_nonseizure_subjects)
    nsz_val_split = int(0.9 * num_nonseizure_subjects)

    sz_train_subjects = seizure_subjects[:sz_train_split]
    sz_val_subjects = seizure_subjects[sz_train_split:sz_val_split]
    sz_test_subjects = seizure_subjects[sz_val_split:]

    nsz_train_subjects = nonseizure_subjects[:nsz_train_split]
    nsz_val_subjects = nonseizure_subjects[nsz_train_split:nsz_val_split]
    nsz_test_subjects = nonseizure_subjects[nsz_val_split:]

    # Convert subject IDs to chunk indices
    # For each subject, get all 23 chunks belonging to that subject
    def get_chunk_indices(subject_ids, chunks_per_subject):
        indices = []
        for subject_id in subject_ids:
            start_idx = subject_id * chunks_per_subject
            end_idx = start_idx + chunks_per_subject
            indices.extend(range(start_idx, end_idx))
        return np.array(indices)

    sz_train_idx = get_chunk_indices(sz_train_subjects, chunks_per_subject)
    sz_val_idx = get_chunk_indices(sz_val_subjects, chunks_per_subject)
    sz_test_idx = get_chunk_indices(sz_test_subjects, chunks_per_subject)

    nsz_train_idx = get_chunk_indices(nsz_train_subjects, chunks_per_subject)
    nsz_val_idx = get_chunk_indices(nsz_val_subjects, chunks_per_subject)
    nsz_test_idx = get_chunk_indices(nsz_test_subjects, chunks_per_subject)

    # Build train/val/test sets
    X_train = np.vstack(
        [
            np.array([SeizureData[i] for i in sz_train_idx], dtype=np.float64),
            np.array([NonSeizureData[i] for i in nsz_train_idx], dtype=np.float64),
        ]
    )
    Y_train = np.hstack(
        [
            np.ones(len(sz_train_idx), dtype=np.int64),
            np.zeros(len(nsz_train_idx), dtype=np.int64),
        ]
    )

    X_val = np.vstack(
        [
            np.array([SeizureData[i] for i in sz_val_idx], dtype=np.float64),
            np.array([NonSeizureData[i] for i in nsz_val_idx], dtype=np.float64),
        ]
    )
    Y_val = np.hstack(
        [
            np.ones(len(sz_val_idx), dtype=np.int64),
            np.zeros(len(nsz_val_idx), dtype=np.int64),
        ]
    )

    X_test = np.vstack(
        [
            np.array([SeizureData[i] for i in sz_test_idx], dtype=np.float64),
            np.array([NonSeizureData[i] for i in nsz_test_idx], dtype=np.float64),
        ]
    )
    Y_test = np.hstack(
        [
            np.ones(len(sz_test_idx), dtype=np.int64),
            np.zeros(len(nsz_test_idx), dtype=np.int64),
        ]
    )

    # Shuffle within each set to mix seizure and non-seizure samples
    train_perm = np.random.permutation(len(X_train))
    val_perm = np.random.permutation(len(X_val))
    test_perm = np.random.permutation(len(X_test))

    X_train, Y_train = X_train[train_perm], Y_train[train_perm]
    X_val, Y_val = X_val[val_perm], Y_val[val_perm]
    X_test, Y_test = X_test[test_perm], Y_test[test_perm]

    # Z-score normalization using training set statistics
    X_train_norm, mean_train, std_train = Normalize(X_train)
    X_val_norm = [(x - mean_train) / std_train for x in X_val]
    X_test_norm = [(x - mean_train) / std_train for x in X_test]

    X_train = np.array(X_train_norm)
    X_val = np.array(X_val_norm)
    X_test = np.array(X_test_norm)

    print("Train: {} samples".format(len(X_train)))
    print("Val: {} samples".format(len(X_val)))
    print("Test: {} samples".format(len(X_test)))

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


if __name__ == "__main__":
    X_train, Y_train, X_val, Y_val, X_test, Y_test = preprocess()
    print("X_train shape:", X_train.shape)

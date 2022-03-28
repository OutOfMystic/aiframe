import numpy as np
from typing import List, Any


def separate_column(table: List[List[Any]], col_num: int):
    """Removes a column from table and returns them separately"""
    columns = [[] for _ in range(len(table[0]))]
    for row in table:
        for i, element in enumerate(row):
            columns[i].append(element)

    separated_col = columns.pop(col_num)

    del table
    new_array = [[] for _ in range(len(columns[0]))]
    for col in columns:
        for i, element in enumerate(col):
            new_array[i].extend(element)

    return new_array, separated_col


def labels_by_reals(inputs: np.ndarray,
                    reals: np.ndarray,
                    labels_count: int,
                    shuffle=True):
    """Replaces source data output reals with labels.

    Every input row index should match the output row index. All real
    values will be replaced with int label value.

    :param inputs: 2-dim numpy array of neurons' input values
    :param reals: 1-dim numpy array of float outputs
    :param labels_count: number of labels into which reals will be divided
    :param shuffle: mix source after sorting. Recommended to set True
    :return: tuple of two numpy arrays with inputs and labels
    """
    vreals = reals.reshape(-1, 1)
    del reals
    stacked_inputs = np.hstack([inputs, vreals])
    del inputs
    sorted_inputs = stacked_inputs[np.argsort(stacked_inputs[:, -1])]
    del stacked_inputs

    res_len = len(sorted_inputs)
    intervals = [int(res_len * (i + 1) / labels_count) for i in range(labels_count)]
    last_interval = 0
    for res_value, interval in enumerate(intervals):
        for res_key in range(last_interval, interval):
            if res_key == last_interval:
                interval_start = sorted_inputs[res_key, -1]
                print(f"Interval {res_value} starts with {interval_start:.2f}", end='')
            if res_key == interval - 1:
                interval_end = sorted_inputs[res_key, -1]
                print(f" and ends with {interval_end:.2f}")
            sorted_inputs[res_key, -1] = res_value
        last_interval = interval
    if shuffle:
        np.random.shuffle(sorted_inputs)

    outputs = sorted_inputs[:, :-1]
    return sorted_inputs[:, -1], outputs.astype('int16')

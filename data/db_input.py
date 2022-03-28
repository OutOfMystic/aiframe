from typing import Sequence, Any, Tuple, List, Union
import numpy as np


def one_hot(column: Sequence[Any]) -> List[List[float]]:
    """Encodes given column using one-hot encoding

    Examples:
    ['a', 'b', 'a'] -> [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
    [0, 2.0] -> [[1.0, 0.0], [0.0, 1.0]]
    """
    classes = []
    new_column = []
    for value in column:
        if value not in classes:
            classes.append(value)
    classes.sort()

    for value in column:
        current_class = classes.index(value)
        new_column.append(current_class)

    result = []
    num_of_classes = len(classes)
    for value in new_column:
        to_result = [0.0 for _ in range(num_of_classes)]
        to_result[value] = 1.0
        result.append(to_result)
    return result


def label(column: Sequence[Union[str, int, float]]) -> List[Tuple[float]]:
    """Encodes given column using label encoding

    Example:
    ['0', '2.0', '2.5'] -> [0.0, 0.8, 1.0]
    """
    column = [float(value) for value in column]
    max_value = max(column)
    min_value = min(column)
    the_range = max_value - min_value

    new_column = [((val - min_value) / the_range,) for val in column]
    return new_column


def reformat(array: Sequence[Sequence[str]],
             column_map: List[str]) -> List[List[float]]:
    # reshaping to column format
    columns = [[] for _ in range(len(array[0]))]
    for row in array:
        for i, element in enumerate(row):
            columns[i].append(element)

    # 'ig' literal
    decrement = 0
    for i, literal in enumerate(column_map.copy()):
        if literal == 'ig':
            del columns[i - decrement]
            del column_map[i - decrement]
            decrement += 1
    # 'la', 'oh', 'cp' literal
    for i, column in enumerate(columns):
        if column_map[i] == 'oh':
            columns[i] = one_hot(column)
        elif column_map[i] == 'la':
            columns[i] = label(column)
        elif column_map[i] == 'cp':
            columns[i] = _floatize(column)
        elif column_map[i] == 'ig':
            continue

    # reshaping back to row format
    del array
    new_array = [[] for _ in range(len(columns[0]))]
    for col in columns:
        for i, element in enumerate(col):
            new_array[i].extend(element)
    return new_array


def _floatize(column):
    return [float(cell) for cell in column]
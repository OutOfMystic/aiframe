import os
import numpy as np
from typing import Sequence, List, Tuple, Union, Callable
from numpy import ndarray

from . import db_input
from . import dir_input
from . import preparing


def load_last(project_name: str) -> Tuple[Tuple[ndarray, ndarray],
                                          Tuple[ndarray, ndarray]]:
    """Quickly loads last input data.

    :param project_name: name of project directory in ``saved``
    :return: prepared source for learning process
    """
    learning_in = np.load(f'saved_inputs\\{project_name}-lin.pkl', allow_pickle=True)
    print('Loaded learning data')
    learning_out = np.load(f'saved_inputs\\{project_name}-lout.pkl', allow_pickle=True)
    training_in = np.load(f'saved_inputs\\{project_name}-tin.pkl', allow_pickle=True)
    training_out = np.load(f'saved_inputs\\{project_name}-tout.pkl', allow_pickle=True)

    return (learning_in, learning_out), (training_in, training_out)


def load_table(array: Sequence[Sequence[str]],
               literals: List[str]):
    """Loads table by parsing each column using available modes.

    Special literals corresponds to available modes
    of parsing. These literals are:

    'oh' - one-hot encoding

    'la' - label encoding (only for int and float cells)

    'ig' - ignore column (some extra data)

    'cp' - just copy cell value to input neuron

    :param array: table of some data
    :param literals: list of parse modes
    :return: table of float values from 0.0 to 1.0
    """
    return db_input.reformat(array, literals)


def load_from_dir(dir_path: str,
                  load_file: Callable = None,
                  output_by_name: Callable = None,
                  max_count: int = -1):
    """Loads lists files and filenames from given directory.

    By default, files are loaded as numpy arrays of real numbers.
    Load procedure can be changed by defining ``load_file`` param. By default,
    filenames are loaded as they are, but they can be parsed by defining
    ``output_by_name`` param.

    :param dir_path: a directory from which the files are loaded
    :param load_file: load file procedure, defaults to None
    :type load_file: function
    :param output_by_name: filename parse procedure, defaults to None
    :type output_by_name: function
    :param max_count: limit of loaded files. If -1, ignored, defaults to -1
    :return: tuple of two lists with formatted file contents and filenames
    """
    return dir_input.reformat(dir_path, load_file, output_by_name, max_count)


def divide(inputs: List[List[float]],
           results: List[Union[int, float]],
           save_name: str = None,
           test_prop=0.1,
           nptype=np.float32):
    """Divides the whole data on learning and training data.

    The propability with which each example is belongs
    to the training group is ``test_prop``.

    :param inputs: table of values of input neurons
    :param results: expected outputs of neural network
    :param save_name: how to name saved project. if None, isn't saved.
    :param test_prop: propability of belong to training data
    :param nptype: returned numpy array type, defaults to ``np.float32``
    :return: prepared source for learning process
    """
    return preparing.divide(inputs, results, save_name, test_prop, nptype)
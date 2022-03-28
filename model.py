import os
from typing import Sequence, Any

import tensorflow.compat.v1 as tf
from tensorflow import keras
from keras.engine.sequential import Sequential


def create_model(lays: Sequence[Any]) -> Sequential:
    """Create model from given sequence of lays

    Lays can be presented as int | str or as complete
    keras layer.
    Input layer may also be a sequence of integers
    representing an input shape.
    """
    parsed_lays = []
    parsed_lays.append(_decode_input_lay(lays[0]))

    for i in range(1, len(lays) - 1):
        parsed_lays.append(_decode_lay((lays[i - 1])))

    if isinstance(lays[-1], str):
        parsed_lays.append(keras.layers.Dense(int(lays[-1]),
                                              activation=tf.nn.softmax))
    if isinstance(lays[-1], int):
        parsed_lays.append(keras.layers.Dense(lays[-1],
                                              activation=tf.nn.softmax))

    return keras.Sequential(parsed_lays)


def save_model(model, name):
    pass


def _decode_input_lay(encoded):
    if hasattr(encoded, '__iter__'):
        return keras.layers.Flatten(input_shape=encoded) # (20, 12)
    elif isinstance(encoded, str):
        if encoded.isnumeric():
            encoded = int(encoded)
            return keras.layers.Input(encoded) # '12'
    elif isinstance(encoded, int):
        return keras.layers.Input(encoded) # 12
    return _decode_lay(encoded)


def _decode_lay(encoded):
    if isinstance(encoded, int):
        return keras.layers.Dense(encoded, activation=tf.nn.relu) # 12
    elif isinstance(encoded, str):
        return keras.layers.Dense(int(encoded), activation=tf.nn.relu) # '12'
    else:
        return encoded # Conv2d(32, (2, 2))
    #raise AssertionError('Lay format should be a string or integer')
    #type, num = separate_type(encoded)


def _separate_type(typestring):
    assert_text = "Lay format incorrect. Examples: 'c3', '21'"
    assert isinstance(typestring[0], str), assert_text

    separator = int()
    for i, char in enumerate(typestring):
        if char.isnumeric():
            separator = i
            break
    else:
        raise AssertionError('String lay format should'
                             ' contain lay type and number of neurons')
    type = typestring[:separator]
    num = typestring[separator:]
    return type, int(num)
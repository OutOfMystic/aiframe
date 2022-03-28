import os
import random
import numpy as np
from PIL import Image


def reformat(dir_path,
             load_file,
             output_by_name,
             max_count):
    counter = 0
    inputs = []
    outputs = []

    if not load_file:
        load_file = _load_file
    if not output_by_name:
        output_by_name = _output_by_name

    for root, _, files in os.walk(dir_path):
        random.shuffle(files)
        all_count = len(files)
        for name in files:
            full_name = os.path.join(root, name)
            output = output_by_name(name)
            if not output:
                continue

            counter += 1
            if counter % 100 == 0:
                percent = counter * 100 // all_count
                print(f'Loading files -> {percent}%')

            content = load_file(full_name)
            inputs.append(content)
            outputs.append(output)
            if counter == max_count:
                break
        if counter == max_count:
            break
    return inputs, outputs


def _load_file(file_name):
    image = Image.open(file_name)
    pix = np.array(image)
    return pix


def _output_by_name(name):
    return name
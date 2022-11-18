import os
import threading
import random
import time

from aiframe.export.solver import Solver
from examples.rnn.input_data.input_data import differences


def translate(words):
    return differences(*words)


def experiment():
    result = solver.solve_pack(['Динамо Спартак', 'Динамо Спартак'] * random.randint(1, 2))
    print(result)


path = os.path.join('saved_models', 'Semantic')
solver = Solver(path, in_converter=translate)
for _ in range(10):
    for _ in range(10):
        threading.Thread(target=experiment).start()
    time.sleep(0.2)
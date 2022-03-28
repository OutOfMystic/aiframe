import os
import random
import numpy as np
from typing import List, Union


def divide(inputs: List[List[float]],
           results: List[Union[int, float]],
           save_name: str = None,
           test_prop=0.1,
           nptype=np.float32):
    learn_inputs = []
    learn_results = []
    test_inputs = []
    test_results = []

    all_count = len(results)
    counter = 0
    while inputs:
        counter += 1
        if test_prop > random.random():
            test_inputs.append(inputs.pop())
            test_results.append(results.pop())
        else:
            learn_inputs.append(inputs.pop())
            learn_results.append(results.pop())
        if counter % 1000 == 0:
            percent = counter * 100 // all_count
            print(f'Dividing -> {percent}%')
    del inputs
    del results

    print('Numping input...')
    learn_inputs_ = np.array(learn_inputs, dtype=nptype)
    learn_results_ = np.array(learn_results, dtype=nptype)
    del learn_inputs
    del learn_results
    test_inputs_ = np.array(test_inputs, dtype=nptype)
    test_results_ = np.array(test_results, dtype=nptype)
    del test_inputs
    del test_results

    if not os.path.exists('saved_inputs'):
        os.mkdir('saved_inputs')
    if save_name:
        print(f"Saving source '{save_name}'...")
        learn_inputs_.dump(f'saved_inputs\\{save_name}-lin.pkl')
        learn_results_.dump(f'saved_inputs\\{save_name}-lout.pkl')
        test_inputs_.dump(f'saved_inputs\\{save_name}-tin.pkl')
        test_results_.dump(f'saved_inputs\\{save_name}-tout.pkl')
    return (learn_inputs_, learn_results_), (test_inputs_, test_results_)
import itertools
import json
import os
import random
import time
from multiprocessing import Queue, Pipe
from threading import Thread

import numpy as np
from loguru import logger
from tensorflow.python import keras

from .items import NumQueue


class Solver(Thread):

    def __init__(self, model_name,
                 in_converter=None,
                 out_converter=None,
                 remember_answers=False,
                 timeout=60):
        super().__init__()
        self.model_name = model_name
        self.remember_answers = remember_answers
        self.in_converter = in_converter if in_converter is not None else np.array
        self.out_converter = out_converter if out_converter is not None else np.array
        self.timeout = timeout

        self.queue = Queue()
        self._model = None
        self._solutions = {}
        self._timeouted = []
        self._answers = {}

        self.start()

    def solve_pack(self, items):
        tupled_items = tuple(items)
        return self.solve(items, is_pack=True)

    def solve(self, item, is_pack=False):
        if not item:
            return tuple()
        hash_ = random.random()
        task = {'hash': hash_, 'item': item, 'is_pack': is_pack}
        self.queue.put(task)

        start_time = time.time()
        while True:
            if (time.time() - start_time) > self.timeout:
                self._timeouted.append(hash_)
                raise RuntimeError(f'Solver stack overflow! {self.timeout} sec timeout has been reached')
            solution = self.get_solution(hash_)
            if solution is None:
                continue
            if not is_pack:
                solution = solution[0]
            return solution

    def get_solution(self, hash_):
        if hash_ not in self._solutions:
            time.sleep(0.1)
        else:
            return self._solutions.pop(hash_)

    def run(self):
        self._init_model()
        while True:
            task = self.queue.get()
            tasks = [task]
            while not self.queue.empty():
                task = self.queue.get()
                tasks.append(task)

            """if self.remember_answers:
                if tupled_items in self._answers:
                    return self._answers[tupled_items]
                else:
                    answers = self.solve(items, is_pack=True)
                    self._answers[tupled_items] = answers
                    return answers"""

            unpack_iters = [self._unpack_task(task) for task in tasks]
            unpacked_tasks = list(itertools.chain(*unpack_iters))
            solutions =
            if self.remember_answers:
                for i, task in enumerate(unpacked_tasks):
                    item = task['item']
                    if item in self._answers:
                        solutions[i] = self._answers[]

            translated = [self.in_converter(task['item']) for task in unpacked_tasks]
            inputs = np.array(translated)
            outputs = self._model.predict(inputs)

            solutions = {}
            for output, task in zip(outputs, unpacked_tasks):
                hash_ = task['hash']
                if hash_ not in solutions:
                    solutions[hash_] = []
                output = self.out_converter(output)
                assert output is not None, 'got ``None`` Answer, use ``False`` instead'
                solutions[hash_].append(output)
            self._solutions.update(solutions)

            for hash_ in self._timeouted:
                del self._solutions[hash_]

    def _init_model(self):
        for _ in range(10):
            try:
                self._model = keras.models.load_model(self.model_name)
            except AttributeError as err:
                print(f'Loading model aerror: {err}')
        asset_path = os.path.join(self.model_name, 'assets', 'summary.json')
        with open(asset_path, 'r') as f:
            summary = json.load(f)
        input_shape = tuple(summary['input_shape'])
        self._model.build(input_shape)

    @staticmethod
    def _unpack_task(task):
        items = []
        if task['is_pack']:
            for item in task['item']:
                new_item = {'hash': task['hash'], 'item': item}
                items.append(new_item)
            return items
        else:
            new_item = {'hash': task['hash'], 'item': task['item']}
            return [new_item]


class PipeSolver(Thread):

    def __init__(self, model_name, translator=None):
        super().__init__()
        self.model_name = model_name
        if translator is None:
            translator = np.array
        self.translator = translator

        self.pipes = {}
        self.queue = NumQueue()
        self._model = None

        self.start()

    def _init_model(self):
        self._model = keras.models.load_model(self.model_name)
        self._model.build()

    def register_pipe(self, key):
        parent_conn, child_conn = Pipe()
        self.pipes[key] = parent_conn
        return child_conn

    def run(self):
        self._init_model()
        while True:
            task = self.queue.get()
            tasks = [task]
            while not self.queue.empty():
                task = self.queue.get()
                tasks.append(task)

            translated = [self.translator(task.data) for task in tasks]
            inputs = np.array(translated)
            outputs = self._model.predict(inputs)
            for output, task in zip(outputs, tasks):
                task.solved(output)
                pipe = self.pipes[task.pipe_key]
                pipe.send(task)

import itertools
import json
import os
import pickle
import random
import time
from multiprocessing import Queue, Pipe
from threading import Thread
from typing import Iterable

import numpy as np
import tensorflow as tf
from loguru import logger
from tensorflow.python import keras

from .items import NumQueue


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
    except RuntimeError as e:
        print(e)


class Solver(Thread):

    def __init__(self, model_name,
                 in_converter=None,
                 out_converter=None,
                 remember_answers=False,
                 timeout=60):
        super().__init__()
        self.model_name = model_name
        self.cache_path = os.path.join(self.model_name, 'assets', 'cache.pkl')
        self.remember_answers = remember_answers
        self.in_converter = in_converter if in_converter is not None else np.array
        self.out_converter = out_converter if out_converter is not None else np.array
        self.timeout = timeout

        self.queue = Queue()
        self._model = None
        self._timeouted = []
        self._solutions = {}
        try:
            with open(self.cache_path, 'rb') as f:
                self._answers = pickle.load(f)
        except:
            self._answers = {}

        self.start()

    def solve_pack(self, items: Iterable):
        if not items:
            return []
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

            unpack_iters = [self._unpack_task(task) for task in tasks]
            unpacked_tasks = list(itertools.chain(*unpack_iters))
            solutions = [None for _ in unpacked_tasks]
            if self.remember_answers:
                for i, task in enumerate(unpacked_tasks):
                    item = task['item']
                    solutions[i] = self._answers.get(item, None)
            tasks_to_solve = {i: task for i, task in enumerate(unpacked_tasks) if solutions[i] is None}

            translated = [self.in_converter(task['item']) for task in tasks_to_solve.values()]
            inputs = np.array(translated)
            outputs = self._model.predict(inputs) if translated else []

            packed_solutions = {}
            for output, index in zip(outputs, tasks_to_solve.keys()):
                output = self.out_converter(output)
                if output is None:
                    logger.warning('got ``None`` Answer from out_converter, use ``False`` instead')
                    output = False
                solutions[index] = output
            for answer, task in zip(solutions, unpacked_tasks):
                hash_ = task['hash']
                if hash_ not in packed_solutions:
                    packed_solutions[hash_] = []
                packed_solutions[hash_].append(answer)
                if self.remember_answers:
                    item = task['item']
                    self._answers[item] = answer
            self._solutions.update(packed_solutions)

            for hash_ in self._timeouted:
                del self._solutions[hash_]
            if self.remember_answers and translated:
                with open(self.cache_path, 'wb+') as f:
                    pickle.dump(self._answers, f)

    def _init_model(self):
        for _ in range(10):
            try:
                self._model = keras.models.load_model(self.model_name)
            except AttributeError as err:
                print(f'Loading model error: {err}')
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

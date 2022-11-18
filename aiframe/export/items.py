import random
from multiprocessing.queues import Queue



class Task:
    def __init__(self, data, pipe_key):
        self.hash = random.random()
        self.data = data
        self.pipe_key = pipe_key

    def solved(self, data):
        self.data = data


class NumQueue(Queue):

    def __init__(self):
        super().__init__()

    def num_put(self, item, pipe_key):
        task = Task(item, pipe_key)
        self.put(task)
        return task.hash

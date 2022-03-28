import os

import tensorflow.compat.v1 as tf
from keras import Model

optimizer_presets = {
    'adam': tf.keras.optimizers.Adam()
}


class Experiment:
    """This is class representation of a simple machine learning
    workspace.

    ``optimizer`` also can be defined with keras optimizers or by one of
    presets. Presets are defined in ``optimizer_presets``.
    You can define options by ``mode`` preset of setting or manually
    configuring following parameters: ``optimizer``, ``loss``, ``metrics``,
    ``epochs``. Mode presets are:

    'classify' - preset for classification task based on Adam (by def.)

    :param model: a ``keras.Model`` that will be compilled and fitted
    :type model: class:`keras.Model`
    :param mode: a set of compile settings. Options are described above
    :type mode: str"""
    def __init__(self, model: Model, mode='classify'):
        """Constructor method
        """
        self._compiled = False
        self.last_test = {}
        self.model = model
        self.optimizer = None
        self.loss = None
        self.metrics = []
        self.epochs = 1
        self._set_mode(mode)

    def _set_mode(self, mode):
        if mode == 'classify':
            self._set_optimizer('adam')
            self.metrics = ['accuracy']
            self.loss = tf.keras.losses.SparseCategoricalCrossentropy()
        else:
            raise ValueError(f'there is no mode with name {mode}')

    def __set__(self, obj, val):
        if obj == 'optimizer':
            self._set_optimizer(val)
        else:
            setattr(self, obj, val)

    def _set_optimizer(self, optimizer):
        if isinstance(optimizer, str):
            self.optimizer = optimizer_presets[optimizer]
        else:
            self.optimizer = optimizer

    def summary(self, input_data):
        """Shows summary for lay structure."""
        if self._compiled:
            self.model.summary()
            return
        test_ins, test_outs = input_data[1]
        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss,
                           metrics=self.metrics)
        self._compiled = True
        self.model.fit(test_ins[:2], test_outs[:2], batch_size=2)
        self.model.summary()

    def run_once(self, input_data):
        """
        Starts learning of ``model`` on ``input_data`` source.
        Input data could be formatted using ``data.divide``

        :param input_data: tuple of tow tuples with train and test data
        :return: tuple of test loss and test accuracy after learning
        """
        (train_ins, train_outs), (test_ins, test_outs) = input_data
        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss,
                           metrics=self.metrics)
        self.model.fit(train_ins, train_outs, epochs=self.epochs)
        self._compiled = True

        predictions = self.model.predict(test_ins)
        self.last_test = {inp: predict for inp, predict in zip(test_ins, predictions)}
        return self.model.evaluate(test_ins, test_outs, verbose=2)

    def save(self, model_name):
        """Saves model state to saved_models directory."""
        if not os.path.exists('saved_models'):
            os.mkdir('saved_models')
        self.model.save(f'saved_models\\{model_name}')
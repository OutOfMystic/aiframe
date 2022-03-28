from __future__ import print_function, division

import os
import time
import json
from functools import partial

import tensorflow.keras.backend as K
from data import parse
import numpy as np
import tensorflow as tf
from PIL import Image
from gan import gan
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from tensorflow.python.keras.layers.merge import _Merge
from keras.models import Sequential, Model
from matplotlib import pyplot, rcParams
from tensorflow.keras.optimizers import RMSprop


class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""

    def _merge_function(self, inputs: list):
        alpha = K.random_uniform((32, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


class WGANGP():

    def __init__(self, noize_shape, img_shape, start_dense=(8, 8, 128), frames=None,
                 restore=False, img_table=(3, 3), name='Project'):
        self.img_shape = img_shape
        self.latent_dim = np.prod(noize_shape)
        self.img_table = img_table
        self.name = name
        self.restore = restore
        self.frames = frames
        self.project_path = os.path.join('projects', self.name)
        self._video_noise = np.random.uniform(-1, 1, (np.prod(img_table), self.latent_dim))

        self.last_epoch = 0
        self.pause = False
        self.to_plot = {}

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        optimizer = RMSprop(learning_rate=0.00005)

        if not os.path.exists('projects'):
            os.mkdir('projects')
        if not os.path.exists('saved_inputs'):
            os.mkdir('saved_inputs')
        if not os.path.exists(os.path.join('projects', self.name)):
            os.mkdir(os.path.join('projects', self.name))
            os.mkdir(os.path.join('projects', self.name, 'video'))
            os.mkdir(os.path.join('projects', self.name, 'images'))
            os.mkdir(os.path.join('projects', self.name, 'combined'))

        # Build the generator and critic
        self.generator = self.build_generator(start_dense)
        self.critic = self.build_critic()

        # -------------------------------
        # Construct Computational Graph
        #       for the Critic
        # -------------------------------

        # Freeze generator's layers while training critic
        self.generator.trainable = False

        # Image input (real sample)
        real_img = Input(shape=self.img_shape)

        # Noise input
        z_disc = Input(shape=(self.latent_dim,))
        # Generate image based of noise (fake sample)
        fake_img = self.generator(z_disc)

        # Discriminator determines validity of the real and fake images
        fake = self.critic(fake_img)
        valid = self.critic(real_img)

        # Construct weighted average between real and fake images
        interpolated_img = RandomWeightedAverage()([real_img, fake_img])
        # Determine validity of weighted sample
        validity_interpolated = self.critic(interpolated_img)

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        partial_gp_loss = partial(self.gradient_penalty_loss,
                                  averaged_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty'  # Keras requires function names

        self.critic_model = Model(inputs=[real_img, z_disc],
                                  outputs=[valid, fake, validity_interpolated])
        self.critic_model.compile(loss=[self.wasserstein_loss,
                                        self.wasserstein_loss,
                                        partial_gp_loss],
                                  optimizer=optimizer,
                                  loss_weights=[1, 1, 10])
        # -------------------------------
        # Construct Computational Graph
        #         for Generator
        # -------------------------------

        # For the generator we freeze the critic's layers
        self.critic.trainable = False
        self.generator.trainable = True

        # Sampled noise for input to generator
        z_gen = Input(shape=(self.latent_dim,))
        # Generate images based of noise
        img = self.generator(z_gen)
        # Discriminator determines validity
        valid = self.critic(img)
        # Defines generator model
        self.generator_model = Model(z_gen, valid)
        self.generator_model.compile(loss=self.wasserstein_loss, optimizer=optimizer)

        self.load_config(restore=self.restore)

    def save_imgs(self, epoch):
        gen_imgs = self.generator.predict(self._video_noise)
        gen_imgs = 127.5 * gen_imgs + 127.5
        np_image = gan.stack_images(gen_imgs, self.img_table, self.img_shape[:2])
        im_image = Image.fromarray(np_image, 'RGB')
        im_image.save(os.path.join("projects", self.name, "images", "%d.png" % epoch))

    def load_config(self, restore=False):
        config_path = os.path.join('projects', self.name, 'config.json')
        try:
            with open(config_path) as f:
                config = json.load(f)
            if restore:
                self.last_epoch = config['last_epoch']
            self.pause = config['pause']
            if self.last_epoch == 0:
                self.pause = False
                self.save_config()
            print('Config loaded')
        except:
            print('Error loading config')

    def log_plot(self, **kwargs):
        if self.last_epoch == 0:
            for kwarg in kwargs:
                self.to_plot[kwarg] == []
        for kwarg in kwargs:
            graph = self.to_plot[kwarg]
            graph.append(kwargs[kwarg])

    def save_config(self):
        config_path = os.path.join('projects', self.name, 'config.json')
        try:
            config = {
                'last_epoch': self.last_epoch,
                'pause': self.pause
            }
            with open(config_path, 'w') as f:
                json.dump(config, f)
            print('Config saved')
        except:
            print('Error saving config')

    def plot_history(self):
        for label in self.to_plot:
            pyplot.plot(self.to_plot[label], label=label)
        pyplot.legend()
        pyplot.savefig(os.path.join(self.project_path, 'plot_stats.png'), dpi=500)
        pyplot.close()
        print('History ploted')

    def manage(self, img_save_interval):
        if self.last_epoch % img_save_interval == 0:
            try:
                self.save_imgs(self.last_epoch)
            except Exception as err:
                print(f'Error saving img: {err}')
        # if self.last_epoch + 1 <= max(self.frames):
        #    try:
        #        self.video_frame()
        #    except Exception as exc:
        #        print(f'Video frame error: {exc}')

        self.last_epoch += 1

        if self.last_epoch % 100 == 0:
            try:
                self.plot_history()
            except:
                print('plot_history error')

        if self.last_epoch % 100 == 0:
            self.load_config(restore=False)

        if self.last_epoch + 1 in self.frames:
            self.save_config()

        while self.pause:
            print('Paused')
            try:
                time.sleep(20)
                self.load_config(restore=False)
            except:
                self.pause = False
                self.save_config()

    def gradient_penalty(self, batch_size, real_images, fake_images):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.critic(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self, start_dense):
        model = Sequential()

        model.add(Dense(np.prod(start_dense), activation="relu", input_dim=self.latent_dim))
        model.add(Reshape(start_dense))

        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(UpSampling2D())
        model.add(Conv2D(112, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(UpSampling2D())
        model.add(Conv2D(96, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(Conv2D(self.img_shape[-1], kernel_size=4, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_critic(self):

        model = Sequential()

        model.add(Conv2D(64, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.15))
        model.add(Conv2D(78, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.15))
        model.add(Conv2D(96, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.15))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.15))
        model.add(Conv2D(144, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.15))
        model.add(Flatten())
        model.add(Dense(1))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train_step(self, real_images):
        if isinstance(real_images, tuple):
            real_images = real_images[0]

        # Get the batch size
        batch_size = tf.shape(real_images)[0]

        # For each batch, we are going to perform the
        # following steps as laid out in the original paper:
        # 1. Train the generator and get the generator loss
        # 2. Train the discriminator and get the discriminator loss
        # 3. Calculate the gradient penalty
        # 4. Multiply this gradient penalty with a constant weight factor
        # 5. Add the gradient penalty to the discriminator loss
        # 6. Return the generator and discriminator losses as a loss dictionary

        # Train the discriminator first. The original paper recommends training
        # the discriminator for `x` more steps (typically 5) as compared to
        # one step of the generator. Here we will train it for 3 extra steps
        # as compared to 5 to reduce the training time.
        for i in range(self.d_steps):
            # Get the latent vector
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim)
            )
            with tf.GradientTape() as tape:
                # Generate fake images from the latent vector
                fake_images = self.generator(random_latent_vectors, training=True)
                # Get the logits for the fake images
                fake_logits = self.discriminator(fake_images, training=True)
                # Get the logits for the real images
                real_logits = self.discriminator(real_images, training=True)

                # Calculate the discriminator loss using the fake and real image logits
                d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, real_images, fake_images)
                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + gp * self.gp_weight

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )

        # Train the generator
        # Get the latent vector
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = self.generator(random_latent_vectors, training=True)
            # Get the discriminator logits for fake images
            gen_img_logits = self.discriminator(generated_images, training=True)
            # Calculate the generator loss
            g_loss = self.g_loss_fn(gen_img_logits)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        return {"d_loss": d_loss, "g_loss": g_loss}

    def train(self, batch_size=64, img_save_interval=50, load=False, packs=3):

        # Load the dataset
        x_train = load_data(self.name, packs, load)

        # Rescale -1 to 1
        x_train = (x_train.astype(np.float32) - 127.5) / 127.5

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))
        dummy = np.zeros((batch_size, 1))  # Dummy gt for gradient penalty
        while True:
            noise = None
            d_loss = None
            for _ in range(self.n_critic):
                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, x_train.shape[0], batch_size)
                imgs = x_train[idx]
                # Sample generator input
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                # Train the critic
                d_loss = self.critic_model.train_on_batch([imgs, noise],
                                                          [valid, fake, dummy])

            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.generator_model.train_on_batch(noise, valid)

            # Plot the progress
            print("%d [D loss: %f] [G loss: %f]" % (self.last_epoch, d_loss[0], g_loss))
            self.log_plot(d_loss=d_loss[0], g_loss=g_loss)
            self.manage(img_save_interval)


def load_file(file_name):
    image = Image.open(file_name)
    img = np.array(image.resize(img_shape[:2]))
    x_train = (img.astype(np.float32) - 127.5) / 127.5
    return x_train


def load_data(name, archives, load):
    data = [None for _ in range(archives)]
    func = try_and_load if load else only_load
    for i in range(archives):
        data[i] = func(name, i)
    x_train = np.concatenate(data)
    for i in range(archives):
        del data[0]
    return x_train


def try_and_load(name, num):
    try:
        (x_train, _), (_, _) = parse.load_last(name + f'_{num}')
    except:
        x_train = only_load(name, num)
    return x_train


def only_load(name, num):
    files = parse.load_from_dir(f'input_images_{num}', load_file=load_file)[0]
    empties = [0 for _ in range(len(files))]
    (x_train, _), (_, _) = parse.divide(files, empties, save_name=name + f'_{num}',
                                        test_prop=0, nptype=np.float32)
    return x_train


if __name__ == '__main__':
    rcParams['figure.figsize'] = (10, 6)
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    frames = [10, 30, 50, 101, 200, 300, 500, 750]
    frames.extend(i * 1000 for i in range(1, 50))

    img_shape = (64, 64, 3)
    #vidsize = (img_size[0] * 3 + 40 * 4, img_size[1] * 3 + 40 * 4)
    fpv = 5000

    wgan = WGANGP(noize_shape=(10, 10), start_dense=(8, 8, 128), img_shape=img_shape,
                  img_table=(8, 8), frames=frames, name='diamonds64', restore=False)
    wgan.train(batch_size=32, img_save_interval=50, load=True)

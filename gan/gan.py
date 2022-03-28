import os
import json
import tensorflow.compat.v1 as tf
import numpy as np
import time
import pickle
from keras import Model, Sequential, backend
from keras.layers import *
from tensorflow.keras.optimizers import RMSprop, Adam
from PIL import Image
from skvideo.io import vwrite

from data import parse


class Gan():
    def __init__(self, noize_shape, frames=None,
                 restore=False, img_table=(3, 3), name='Project'):
        self.img_size = img_size
        self.img_rows, self.img_cols = img_size
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.noize_shape = noize_shape
        self._video_noise = np.random.uniform(-1, 1, (np.prod(img_table), np.prod(self.noize_shape)))
        self.frames = frames
        #self.video = np.zeros((min(max(self.frames), fpv), *vidsize, 3), np.uint8)
        self.num = 0
        self.img_table = img_table
        self.restore = restore
        self.name = name
        self.last_epoch = 0
        self.pause = False

        if not os.path.exists(f'projects\\{self.name}'):
            os.mkdir(f'projects\\{self.name}')
            os.mkdir(f'projects\\{self.name}\\video')
            os.mkdir(f'projects\\{self.name}\\images')
            os.mkdir(f'projects\\{self.name}\\combined')

        self.optimizer_disc = Adam(learning_rate=0.0001)
        self.optimizer_gen = Adam(learning_rate=0.0001)
        self.optimizer_multi = Adam(learning_rate=0.0001)
        #self.optimizer_disc = RMSprop(0.0001)
        #self.optimizer_gen = RMSprop(0.0001)
        #self.optimizer_multi = RMSprop(0.0001)

        # Build and compile the generator
        if self.restore:
            print(f'{self.name} generator model RESTORING...')
            generator_path = f'projects\\{self.name}\\generator'
            self.generator = tf.keras.models.load_model(generator_path)
        else:
            self.generator = self.define_generator()
            self.generator.compile(loss='binary_crossentropy', optimizer=self.optimizer_gen)
        self.generator.summary()

        # Build and compile the discriminator
        if self.restore:
            print(f'{self.name} discriminator model RESTORING...')
            discriminator_path = f'projects\\{self.name}\\discriminator'
            self.discriminator = tf.keras.models.load_model(discriminator_path)
        else:
            self.discriminator = self.build_discriminator()
            self.discriminator.compile(loss='binary_crossentropy',
                                       optimizer=self.optimizer_disc,
                                       metrics=['accuracy'])
        self.discriminator.summary()

        # The generator takes noise as input and generated imgs
        noize_input = (np.prod(self.noize_shape),)
        z = Input(shape=noize_input)
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        self.combined = Model(z, valid)
        self.load_config(restore=self.restore)
        if self.restore:
            self.restore_net()
        self.combined.compile(loss='binary_crossentropy', optimizer=self.optimizer_multi)

    def save_net(self):
        generator_path = f'projects\\{self.name}\\generator'
        discriminator_path = f'projects\\{self.name}\\discriminator'
        combined_path = f'projects\\{self.name}\\combined'
        print('Saving generator...')
        self.generator.save(generator_path)
        print('Generator saved. Saving discriminator...')
        self.discriminator.save(discriminator_path)
        self.combined.save_weights(f'projects\\{self.name}\\combined\\weights.h5')
        save_optimizer_state(self.combined.optimizer, combined_path, 'weights.npy')
        print('Discriminator and combined saved')

    def restore_net(self):
        print(f'{self.name} combined model RESTORING...')
        combined_path = f'projects\\{self.name}\\combined\\'
        self.combined.load_weights(combined_path + 'weights.h5')

        grad_vars = self.combined.trainable_weights
        load_optimizer_state(self.optimizer_multi, combined_path[:-1],
                             'weights', grad_vars)
        # self.combined._make_train_function()
        # with open(combined_path + 'optimizer.pkl', 'rb') as f:
        #    weight_values = pickle.load(f)
        # self.combined.optimizer.set_weights(weight_values)

    def load_config(self, restore=False):
        config_path = f'projects\\{self.name}\\config.json'
        try:
            with open(config_path) as f:
                config = json.load(f)
            if restore:
                self.last_epoch = config['last_epoch']
            self.pause = config['pause']
            if self.num == 0:
                self.pause = False
                self.save_config()
            print('Config loaded')
        except:
            print('Error loading config')

    def save_config(self):
        config_path = f'projects\\{self.name}\\config.json'
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

    def train(self, batch_size=128, save_interval=50, inspect=True, load=False):

        # Load the dataset
        x_train = load_data(self.name, 3, load)

        half_batch = int(batch_size / 2)

        while True:
            self.last_epoch += 1
            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(-1, x_train.shape[0], half_batch)
            imgs = x_train[idx]

            noise = np.random.uniform(-1, 1, (half_batch, np.prod(self.noize_shape)))

            # Generate a half batch of new images
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.uniform(-1, 1, (half_batch, np.prod(self.noize_shape)))

            # The generator wants the discriminator to label the generated samples
            # as valid (ones)
            valid_y = np.array([1] * half_batch)

            # Train the generator
            g_loss_1 = self.combined.train_on_batch(noise, valid_y)

            noise = np.random.uniform(-1, 1, (half_batch, np.prod(self.noize_shape)))
            g_loss_2 = self.combined.train_on_batch(noise, valid_y)
            g_loss = (g_loss_1 + g_loss_2) / 2

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (self.last_epoch, d_loss[0],
                                                                  100 * d_loss[1], g_loss))

            # Inspect values
            if inspect:
                gen_labels = self.combined.predict(noise)
                test_part = gen_imgs[:, self.img_rows//2, :]
                to_log = [self.last_epoch, d_loss[0], 100 * d_loss[1], g_loss,
                          np.average(test_part), np.var(test_part),
                          np.average(gen_labels), np.var(gen_labels)]
                lprint(to_log)

            # If at save interval => save generated image samples
            if (self.last_epoch - 1) % save_interval == 0:
                try:
                    self.save_imgs(self.last_epoch - 1)
                except Exception as err:
                    print(f'Error saving img {err}')
            if self.num + 1 <= max(self.frames):
                try:
                    #self.video_frame()
                    if self.num % 1000 == 0:
                        self.save_net()
                except Exception as exc:
                    print(f'Video frame error: {exc}')
            self.num += 1

            if self.num + 1 in self.frames:
                self.save_config()

            if self.num % 100 == 0:
                self.load_config(restore=False)
            while self.pause:
                print('Paused')
                try:
                    time.sleep(20)
                    self.load_config(restore=False)
                except:
                    self.pause = False
                    self.save_config()

    def video_frame(self):
        gen_imgs = self.generator.predict(self._video_noise)
        gen_imgs = 127.5 * gen_imgs + 127.5
        gen_imgs = gen_imgs.astype(np.uint8)
        frame = stack_images(gen_imgs, self.img_table, self.img_size[:2])
        frame_num = self.num % fpv
        video_num = self.num // fpv
        #self.video[frame_num] = frame
        if self.num + 1 in self.frames:
            make_video(video_num, frame_num)

    def save_imgs(self, epoch):
        gen_imgs = self.generator.predict(self._video_noise)
        gen_imgs = 127.5 * gen_imgs + 127.5
        np_image = stack_images(gen_imgs, self.img_table, self.img_size[:2])
        im_image = Image.fromarray(np_image, 'RGB')
        im_image.save(f"projects\\{self.name}\\images\\%d.png" % epoch)

    def define_generator(self):
        model = Sequential()
        n_nodes = [24, 24, 8]

        model.add(Dense(np.prod(n_nodes), input_dim=np.prod(self.noize_shape), use_bias=False))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape(n_nodes))

        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(140, (3, 3), padding="same", use_bias=False))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(128, (3, 3), padding="same", use_bias=False))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        #model.add(UpSampling2D((2, 2)))
        #model.add(Conv2D(112, (3, 3), padding="same", use_bias=False))
        #model.add(BatchNormalization(momentum=0.8))
        #model.add(LeakyReLU(alpha=0.2))

        #model.add(UpSampling2D((2, 2)))
        #model.add(Conv2D(96, (3, 3), padding="same", use_bias=False))
        #model.add(BatchNormalization(momentum=0.8))
        #model.add(LeakyReLU(alpha=0.2))

        #model.add(UpSampling2D((2, 2)))
        #model.add(Conv2DTranspose(78, (3, 3), padding="same", use_bias=False))
        #model.add(BatchNormalization(momentum=0.8))
        #model.add(LeakyReLU(alpha=0.2))

        #model.add(UpSampling2D((2, 2)))
        #model.add(Conv2DTranspose(78, (3, 3), padding="same", use_bias=False))
        #model.add(BatchNormalization(momentum=0.8))
        #model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2DTranspose(3, (5, 5), activation='tanh', padding="same", use_bias=False))

        model.summary()

        noise = Input(shape=(np.prod(self.noize_shape),))
        img = model(noise)

        return Model(noise, img)

    def build_generator(self):
        noise_shape = (np.prod(self.noize_shape),)

        model = Sequential()

        model.add(Dense(6*6*8, input_shape=noise_shape, use_bias=False))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(0.2))
        model.add(Reshape([6, 6, 8]))

        model.add(Conv2D(128, (3, 3), padding="same", use_bias=False))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(0.2))
        #model.add(Dropout(0.4))
        model.add(UpSampling2D((2, 2)))

        model.add(Conv2D(128, (3, 3), padding="same", use_bias=False))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(0.2))
        #model.add(Dropout(0.4))
        model.add(UpSampling2D((2, 2)))

        model.add(Conv2D(112, (3, 3), padding="same", use_bias=False))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(0.2))
        #model.add(Dropout(0.4))
        model.add(UpSampling2D((2, 2)))

        model.add(Conv2D(112, (3, 3), padding="same", use_bias=False))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(0.2))
        #model.add(Dropout(0.4))
        model.add(UpSampling2D((2, 2)))

        model.add(Conv2D(96, (3, 3), padding="same", use_bias=False))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(0.2))
        #model.add(Dropout(0.2))
        model.add(UpSampling2D((2, 2)))

        model.add(Conv2D(96, (3, 3), padding="same", use_bias=False))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(0.2))
        #model.add(Dropout(0.2))
        model.add(UpSampling2D((2, 2)))

        model.add(Conv2D(78, (3, 3), padding="same", use_bias=False))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(0.2))
        #model.add(Dropout(0.2))
        #model.add(UpSampling2D((2, 2)))

        model.add(Conv2D(3, (3, 3), activation='tanh', padding="same", use_bias=False))

        model.summary()

        noise = Input(shape=noise_shape)
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):
        img_shape = (self.img_rows, self.img_cols, self.channels)

        model = Sequential()

        #model.add(Conv2D(96, (3, 3), padding="same"))
        #model.add(LeakyReLU(0.2))
        #model.add(Dropout(0.4))
        #model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(96, (4, 4), input_shape=img_shape, padding="same"))
        model.add(LeakyReLU(0.2))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(112, (4, 4), padding="same"))
        model.add(LeakyReLU(0.2))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.15))

        model.add(Conv2D(112, (4, 4), padding="same"))
        model.add(LeakyReLU(0.2))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.15))

        model.add(Conv2D(128, (4, 4), padding="same"))
        model.add(LeakyReLU(0.2))
        #model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.15))

        #model.add(Conv2DTranspose(140, (4, 4), strides=2, padding="same"))
        #model.add(LeakyReLU(0.2))
        #model.add(Dropout(0.25))
        #model.add(MaxPooling2D((2, 2)))

        model.add(Flatten())

        model.add(Dense(128))
        model.add(LeakyReLU())
        model.add(Dropout(0.35))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(LeakyReLU())
        model.add(Dropout(0.35))

        model.add(Dense(32))
        model.add(LeakyReLU())
        #model.add(MaxPooling2D((2, 2)))
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=img_shape)
        validity = model(img)

        return Model(img, validity)


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


def lprint(to_print):
    if not isinstance(to_print, list) and not isinstance(to_print, tuple):
        to_print = [str(to_print).replace('.', ',')]
    str_generator = (str(elem).replace('.', ',') for elem in to_print)
    formatted = ';'.join(str_generator) + '\n'
    with open('log.csv', 'a') as f:
        f.write(formatted)


def stack_images(arrays, img_table, img_shape, pad=15):
    hor, vert = img_table
    width, height = img_shape
    hor_size = hor * height + pad * (hor + 1)
    vert_size = vert * width + pad * (vert + 1)
    image = np.ones((hor_size, vert_size, 3), np.uint8) * 255
    i = 0
    for x in range(pad, hor_size - height, height + pad):
        for y in range(pad, vert_size - width, width + pad):
            image[x:x+height, y:y+width] = arrays[i]
            i += 1
    return image


def make_video(video_name, frame_num):
    print('Saving video...')
    outputdict = {
        '-b': '3000000'
    }
    #vwrite(f"projects\\{self.name}\\video\\output_{video_name}.mp4", self.video[:frame_num + 1], outputdict=outputdict)
    print('Video saved')


def load_file(file_name):
    image = Image.open(file_name)
    img = np.array(image.resize(img_size))
    x_train = (img.astype(np.float32) - 127.5) / 127.5
    return x_train


def save_optimizer_state(optimizer, save_path, save_name):
    '''
    Save keras.optimizers object state.
save_imgs
    Arguments:
    optimizer --- Optimizer object.
    save_path --- Path to save location.
    save_name --- Name of the .npy file to be created.

    '''
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(os.path.join(save_path, save_name), optimizer.get_weights())
    return


def load_optimizer_state(optimizer, load_path, load_name, model_train_vars):
    '''
    Loads keras.optimizers object state.

    Arguments:
    optimizer --- Optimizer object to be loaded.
    load_path --- Path to save location.
    load_name --- Name of the .npy file to be read.
    model_train_vars --- List of model variables (obtained using Model.trainable_variables)

    '''

    # Load optimizer weights
    opt_weights = np.load(os.path.join(load_path, load_name) + '.npy', allow_pickle=True)
    zero_grads = [tf.zeros_like(w) for w in model_train_vars]
    saved_vars = [tf.identity(w) for w in model_train_vars]
    optimizer.apply_gradients(zip(zero_grads, model_train_vars))
    [x.assign(y) for x, y in zip(model_train_vars, saved_vars)]

    # Set the weights of the optimizer
    optimizer.set_weights(opt_weights)

    return


if __name__ == '__main__':
    grid = 6, 6

    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    save_interval = 1000
    frames = [10, 30, 50, 101, 200, 300, 500, 750]
    frames.extend(i * save_interval for i in range(1, 50))

    img_size = (96, 96)
    vidsize = (img_size[0] * grid[0] + 15 * (grid[0]+1), img_size[1] * grid[1] + 15 * (grid[1]+1))
    fpv = 5000

    gan = Gan(noize_shape=(10, 10, 1), frames=frames, img_table=grid,
              name='diamonds96', restore=False)
    gan.train(batch_size=32, save_interval=50, load=True)

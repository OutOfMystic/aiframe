import os
import sys
import time
import json

s_path = os.path.dirname(os.getcwd())
sys.path.insert(0, s_path)
from data import parse
from gan import gan
from experiment import experiment

from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.optimizers as optimizers
from keras import layers
from matplotlib import pyplot, rcParams


def conv_block(x,
               filters,
               activation,
               kernel_size=(3, 3),
               strides=(2, 2),
               padding="same",
               use_bias=True,
               use_bn=False,
               use_dropout=False,
               use_pooling=True,
               drop_value=0.5):
    if use_pooling:
        x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(filters, kernel_size, strides=strides,
                      padding=padding, use_bias=use_bias)(x)
    if use_bn:
        x = layers.BatchNormalization()(x)
    x = activation(x)
    if use_dropout:
        x = layers.Dropout(drop_value)(x)
    return x


def upsample_block(x,
                   filters,
                   activation,
                   kernel_size=(3, 3),
                   strides=(1, 1),
                   up_size=(2, 2),
                   padding="same",
                   use_bn=False,
                   use_bias=True,
                   use_dropout=False,
                   drop_value=0.3,
                   transp_conv=False):
    x = layers.UpSampling2D(up_size)(x)
    layer = layers.Conv2DTranspose if transp_conv else layers.Conv2D
    x = layer(
        filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias
    )(x)
    
    if use_bn:
        x = layers.BatchNormalization()(x)
    
    if activation:
        x = activation(x)
    if use_dropout:
        x = layers.Dropout(drop_value)(x)
    return x


class WGAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim,
                 discriminator_extra_steps=3,
                 gp_weight=10.0, img_save_interval=50):
        super(WGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight
    
        self.img_save_interval = img_save_interval
        self.latent_dim = latent_dim
    
    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(WGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn
    
    def train_step(self, real_images):
        if isinstance(real_images, tuple):
            real_images = real_images[0]
        
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
                fake_images = self.generator(random_latent_vectors, training=True)
                fake_logits = self.discriminator(fake_images, training=True)
                real_logits = self.discriminator(real_images, training=True)
                
                d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
                gp = self.gradient_penalty(batch_size, real_images, fake_images)
                d_loss = d_cost + gp * self.gp_weight
            
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )
        
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            generated_images = self.generator(random_latent_vectors, training=True)
            gen_img_logits = self.discriminator(generated_images, training=True)
            g_loss = self.g_loss_fn(gen_img_logits)
        
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        kwargs = {"d_loss": d_loss, "g_loss": g_loss}
        return kwargs
    
    def gradient_penalty(self, batch_size, real_images, fake_images):
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff
        
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.discriminator(interpolated, training=True)
        
        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp
    
    def train(self, batch_size=32, project='Project', packs=3, load=False,
              cbk=None, test=False):
        if test:
            mnist = keras.datasets.mnist
            train_images = mnist.load_data()[0][0]
            img_shape = (28, 28, 1)
            train_images = train_images.reshape(train_images.shape[0], *img_shape).astype("float32")
            train_images = (train_images - 127.5) / 127.5
        else:
            train_images = load_data(project, packs, load)
        print(f"Number of examples: {len(train_images)}")
        print(f"Shape of the images in the dataset: {train_images.shape[1:]}")

        """
        length = train_images.shape[0]
        batches = []
        for i in range(0, length - batch_size, batch_size):
            batch = train_images[i:i+batch_size]
            batches.append(batch)
        del train_images
        batch_len = len(batches)
        """

        #start_time = time.time()
        num = -1
        while True:
            idx = np.random.randint(-1, len(train_images), batch_size)
            real_images = train_images[idx]
            #num += 1
            #real_images = batches[num % batch_len]
            
            for i in range(self.d_steps):
                # Get the latent vector
                random_latent_vectors = tf.random.normal(
                    shape=(batch_size, self.latent_dim)
                )
                with tf.GradientTape() as tape:
                    fake_images = self.generator(random_latent_vectors, training=True)
                    fake_logits = self.discriminator(fake_images, training=True)
                    real_logits = self.discriminator(real_images, training=True)
                    
                    d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
                    gp = self.gradient_penalty(batch_size, real_images, fake_images)
                    d_loss = d_cost + gp * self.gp_weight
                
                d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
                self.d_optimizer.apply_gradients(
                    zip(d_gradient, self.discriminator.trainable_variables)
                )
            
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
            with tf.GradientTape() as tape:
                generated_images = self.generator(random_latent_vectors, training=True)
                gen_img_logits = self.discriminator(generated_images, training=True)
                g_loss = self.g_loss_fn(gen_img_logits)
            
            gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
            self.g_optimizer.apply_gradients(
                zip(gen_gradient, self.generator.trainable_variables)
            )
            logs = {"d_loss": d_loss, "g_loss": g_loss}
            del random_latent_vectors
            del generated_images
            del gen_img_logits
            del real_images
            del gen_gradient
            del gp
            if cbk:
                cbk.on_batch_end(cbk.last_epoch, logs)
                print(f'{cbk.last_epoch}: [d_loss: {d_loss:.2f}, g_loss: {g_loss:.2f}]')


class GANMonitor(keras.callbacks.Callback):
    last_epoch = 0
    pause = False
    to_plot = {}
    
    def __init__(self, model, img_shape, img_table=(5, 5), name='Project',
                 latent_dim=128, img_save_interval=50):
        self.model = model
        self.img_shape = img_shape
        self.img_table = img_table
        self.name = name
        self.latent_dim = latent_dim
        self.img_save_interval = img_save_interval
        self.start_time = time.time()
        self._video_noise = np.random.uniform(-1, 1, (np.prod(img_table), self.latent_dim))

        self.project_path = os.path.join('projects', self.name)
        if not os.path.exists('projects'):
            os.mkdir('projects')
        if not os.path.exists('saved_inputs'):
            os.mkdir('saved_inputs')
        if not os.path.exists(os.path.join('projects', self.name)):
            os.mkdir(os.path.join('projects', self.name))
            os.mkdir(os.path.join('projects', self.name, 'video'))
            os.mkdir(os.path.join('projects', self.name, 'images'))
            os.mkdir(os.path.join('projects', self.name, 'combined'))
    
    def log_plot(self, **kwargs):
        if self.last_epoch == 0:
            for kwarg in kwargs:
                self.to_plot[kwarg] = []
        for kwarg in kwargs:
            graph = self.to_plot[kwarg]
            graph.append(kwargs[kwarg])
    
    def plot_history(self):
        approx = 40
        new_plot = {}
        for label in self.to_plot:
            graph = self.to_plot[label]
            new_graph = []
            for i in range(approx - 1, len(graph)):
                average = np.average(graph[i-14:i+1])
                new_graph.append(average)
            new_plot[label] = new_graph
        
        for label in new_plot:
            graph = new_plot[label]
            pyplot.plot(graph, label=label)
        
        pyplot.legend()
        pyplot.savefig(os.path.join(self.project_path, 'plot_stats.png'), dpi=500)
        pyplot.close()
        print('Hp', end='')
    
    def save_imgs(self, epoch):
        gen_imgs = self.model.generator.predict(self._video_noise)
        gen_imgs = 127.5 * gen_imgs + 127.5
        np_image = gan.stack_images(gen_imgs, self.img_table, self.img_shape[:2])
        im_image = Image.fromarray(np_image, 'RGB')
        im_image.save(os.path.join("projects", self.name, "images", "%d.png" % epoch))
        del im_image
    
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
            if restore:
                print('Config loaded')
        except:
            print('Error loading config')
    
    def save_config(self):
        config_path = os.path.join('projects', self.name, 'config.json')
        try:
            config = {
                'last_epoch': self.last_epoch,
                'pause': self.pause
            }
            with open(config_path, 'w') as f:
                json.dump(config, f)
            print('Cs', end='')
        except:
            print('Error saving config')
    
    def manage(self):
        if self.last_epoch % self.img_save_interval == 0:
            try:
                self.save_imgs(self.last_epoch)
            except Exception as err:
                print(f'Error saving img: {err}')
        
        self.last_epoch += 1
        # if self.last_epoch <= max(self.frames):
        #    try:
        #        self.video_frame()
        #    except Exception as exc:
        #        print(f'Video frame error: {exc}')
        if self.last_epoch % 500 == 0:
            try:
                self.plot_history()
            except:
                print('plot_history error')
        if self.last_epoch % 500 == 0:
            try:
                self.save_net()
            except:
                print('MODEL SAVING ERROR')
        if self.last_epoch % 100 == 0:
            self.load_config(restore=False)
        if self.last_epoch % 100 == 0:
            self.save_config()
        
        while self.pause:
            print('Paused')
            try:
                time.sleep(20)
                self.load_config(restore=False)
            except:
                self.pause = False
                self.save_config()
    
    def on_batch_end(self, batch, logs):
        self.log_plot(**logs)
        self.manage()
    
    def save_net(self):
        generator_path = os.path.join('project', self.name, 'generator')
        discriminator_path = os.path.join('project', self.name, 'discriminator')
        print('Saving generator...')
        self.model.generator.save(generator_path)
        print('Generator saved. Saving discriminator...')
        self.model.discriminator.save(discriminator_path)
        print('Discriminator saved')


def save_optimizer_state(optimizer, save_path, save_name):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(os.path.join(save_path, save_name), optimizer.get_weights())
    return


def discriminator_loss(real_img, fake_img):
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    return fake_loss - real_loss


def generator_loss(fake_img):
    return -tf.reduce_mean(fake_img)


def load_file(file_name):
    global img_shape
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


def get_discriminator_model(img_shape, strides=True):
    img_input = layers.Input(shape=img_shape)
    # Zero pad the input to make the input images size to (32, 32, 1).
    stride_value = 2 if strides else 1
    x = conv_block(img_input, 64, layers.LeakyReLU(0.2), 3, stride_value,
                   use_bias=True, use_dropout=False,
                   drop_value=0.3, use_pooling=not strides)
    x = conv_block(x, 64, layers.LeakyReLU(0.2),3,stride_value,
                   use_bias=True,use_dropout=True,
                   drop_value=0.1, use_pooling=not strides)
    x = conv_block(x, 64, layers.LeakyReLU(0.2), 3, stride_value,
                   use_bias=True, use_dropout=True,
                   drop_value=0.1, use_pooling=not strides)
    x = conv_block(x, 96, layers.LeakyReLU(0.2), 3, stride_value,
                   use_bias=True, use_dropout=True,
                   drop_value=0.1, use_pooling=not strides)
    x = conv_block(x, 128, layers.LeakyReLU(0.2), 3, stride_value,
                   use_bias=True, use_dropout=False,
                   drop_value=0.3, use_pooling=not strides)
    
    x = layers.Flatten()(x)
    x = layers.Dense(64)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.15)(x)
    
    x = layers.Dense(32)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.15)(x)
    
    x = layers.Dense(16)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.15)(x)
    
    x = layers.Dense(1)(x)
    
    d_model = keras.models.Model(img_input, x, name="discriminator")
    return d_model


def get_generator_model(noise_dim, start_dense, dimensions, transp_conv=False):
    noise = layers.Input(shape=(noise_dim,))
    x = layers.Dense(np.prod(start_dense), use_bias=False)(noise)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    x = layers.Reshape(start_dense)(x)
    x = upsample_block(x, 164, layers.LeakyReLU(0.2), use_bias=False, use_bn=True, transp_conv=transp_conv)
    x = upsample_block(x, 128, layers.LeakyReLU(0.2), use_bias=False, use_bn=True, transp_conv=transp_conv)
    x = upsample_block(x, 96, layers.LeakyReLU(0.2), use_bias=False, use_bn=True, transp_conv=transp_conv)
    x = upsample_block(x, 64, layers.LeakyReLU(0.2), use_bias=False, use_bn=True, transp_conv=transp_conv)
    x = upsample_block(x, dimensions, layers.Activation("tanh"), kernel_size=3,
                       strides=(1, 1), use_bias=False, use_bn=True, transp_conv=transp_conv)
    
    g_model = keras.models.Model(noise, x, name="generator")
    return g_model


def start_frame(project='Project', batch_size=64,
                img_shape=(96, 96, 3), gpu=True, fit=False,
                noise_dim=100, start_dense=(3, 3, 256),
                packs=1, img_save_table=(5, 5),
                img_save_interval=50, epochs=300,
                strides=False, transp_conv=False,
                learning_rate=0.00005, test=False,
                load=False, adam=False):
    img_table = img_save_table
    
    train_images = []
    if test:
        mnist = keras.datasets.mnist
        train_images = mnist.load_data()[0][0]
        img_shape = (28, 28, 1)
        train_images = train_images.reshape(train_images.shape[0], *img_shape).astype("float32")
        train_images = (train_images - 127.5) / 127.5

    rcParams['figure.figsize'] = (10, 6)
    if gpu:
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if adam:
        generator_optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        discriminator_optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        generator_optimizer = optimizers.RMSprop(learning_rate=learning_rate)
        discriminator_optimizer = optimizers.RMSprop(learning_rate=learning_rate)

    d_model = get_discriminator_model(img_shape, strides=strides)
    d_model.summary()
    g_model = get_generator_model(noise_dim, start_dense, img_shape[-1], transp_conv=transp_conv)
    g_model.summary()

    wgan = WGAN(
        discriminator=d_model,
        generator=g_model,
        latent_dim=noise_dim,
        discriminator_extra_steps=3,
    )

    wgan.compile(
        d_optimizer=discriminator_optimizer,
        g_optimizer=generator_optimizer,
        g_loss_fn=generator_loss,
        d_loss_fn=discriminator_loss,
    )

    cbk = GANMonitor(wgan, img_shape, img_table=img_table, name=project,
                     latent_dim=noise_dim, img_save_interval=img_save_interval)
    if fit:
        if not test:
            train_images = load_data(project, packs, load)
        print('FITTING')
        wgan.fit(train_images, batch_size=batch_size, epochs=epochs, callbacks=[cbk])
    else:
        wgan.train(batch_size=batch_size, project=project, packs=packs, load=load,
                   test=test, cbk=cbk)


if __name__ == '__main__':
    args = {
        "learning_rate": [0.0005, 0.0002, 0.00005],
        "noise_dim": [50, 200],
        "start_dense": [(3, 3, 512), (3, 3, 128), (6, 6, 192)],
        "strides": [True, False],
        "transp_conv": [True, False],
        "project": "diamonds96",
        "packs": 4,
        "batch_size": 32,
        "epochs": 400,
        "fit": True,
        "load": False,
        "img_save_interval": 100,
        "gpu": False
    }
    global img_shape
    img_shape = (96, 96, 3)
    experiments = experiment(args)
    for kwargs in experiments:
        start_frame(**kwargs)
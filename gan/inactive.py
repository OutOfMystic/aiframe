import numpy as np
from keras import Model
from keras.layers import *
from keras.engine.sequential import Sequential

class A:
    def __init__(self, noize_shape, frames=None, zero=True, img_table=(3, 3)):
        self.img_rows, self.img_cols = (100, 100)
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.noize_shape = noize_shape
        self.zero = zero
        self._zero = 0 if zero else -1
        self._video_noise = np.random.uniform(self._zero, 1, (5 * 5, np.prod(self.noize_shape)))
        self.frames = frames
        self.num = 0
        self.img_table = img_table


    def _build_generator(self):

        noise_shape = (256,)

        model = Sequential()

        model.add(Dense(256, input_shape=noise_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=noise_shape)
        img = model(noise)

        return Model(noise, img)

    def _build_discriminator(self):
        img_shape = (self.img_rows, self.img_cols, self.channels)

        model = Sequential()

        model.add(Flatten(input_shape=img_shape))
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.8))
        model.add(Dense(64))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=img_shape)
        validity = model(img)

        return Model(img, validity)

    def __build_discriminator(self):
        img_shape = (self.img_rows, self.img_cols, self.channels)

        model = Sequential()

        model.add(Flatten(input_shape=img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=img_shape)
        validity = model(img)

        return Model(img, validity)

    def _build_generator(self):

        noise_shape = (np.prod(self.noize_shape),)

        inputs = Input(shape=noise_shape)
        x = Reshape(self.noize_shape)(inputs)
        for filters in [128, 92, 64, 48]:
            """x = Activation("relu")(x)
            x = Conv2D(filters, (3, 3), padding="same")(x)
            x = BatchNormalization()(x)"""

            x = Activation('tanh')(x)
            x = Conv2DTranspose(filters, (3, 3), padding="same")(x)
            x = BatchNormalization(momentum=0.8)(x)

            x = UpSampling2D((2, 2))(x)
        if self.zero:
            activation = 'sigmoid'
        else:
            activation = 'tanh'
        outputs = Conv2DTranspose(3, (3, 3), activation=activation, padding='same')(x)
        return Model(inputs, outputs)

    def _build_discriminator(self):
        img_shape = (self.img_rows, self.img_cols, self.channels)

        inputs = Input(shape=img_shape)
        x = Conv2D(24, (3, 3), strides=2, padding="same")(inputs)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        # Blocks 1, 2, 3 are identical apart from the feature depth.
        for filters in [24, 32, 48]:
            x = Activation("relu")(x)
            x = SeparableConv2D(filters, (3, 3), padding="same")(x)
            x = BatchNormalization()(x)

            x = Activation("relu")(x)
            x = SeparableConv2D(filters, (3, 3), padding="same")(x)
            x = BatchNormalization()(x)

            x = MaxPooling2D((2, 2), padding="same")(x)
            #x = Dropout(0.3)(x)
        #x = Dropout(0.5)(x)
        x = Flatten()(x)
        x = Dense(128)(x)
        x = LeakyReLU(alpha=0.2)(x)
        outputs = Dense(1, activation='sigmoid')(x)

        return Model(inputs, outputs)

    def build_generator(self):

        noise_shape = (256,)

        model = Sequential()

        model.add(Input(shape=noise_shape))
        model.add(Reshape(self.noize_shape))
        model.add(Activation("tanh"))
        model.add(Conv2DTranspose(32, (3, 3), padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2DTranspose(64, (3, 3), padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2DTranspose(92, (3, 3), padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D((2, 2)))
        """
        model.add(Flatten())
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(2048))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        """
        #model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=noise_shape)
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):
        img_shape = (self.img_rows, self.img_cols, self.channels)

        model = Sequential()

        #model.add(SeparableConv2D(48, (3, 3), input_shape=img_shape, padding="same"))
        #model.add(MaxPooling2D((2, 2), padding="same"))
        model.add(Flatten(input_shape=img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.8))
        model.add(Dense(64))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=img_shape)
        validity = model(img)

        return Model(img, validity)
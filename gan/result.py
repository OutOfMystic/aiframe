import tensorflow as tf
from stats import backtesting
import numpy as np
from PIL import Image


def stack_images(arrays, width, height, vert=3, hor=3, pad=40):
    hor_size = hor * width + pad * (hor + 1)
    vert_size = vert * height + pad * (vert + 1)
    image = np.ones((hor_size, vert_size, 3), np.uint8) * 255
    i = 0
    for x in range(pad, hor_size - width, width + pad):
        for y in range(pad, vert_size - height, height + pad):
            image[x:x + width, y:y + height, :] = arrays[i]
            i += 1
    return image


def prepare_noise(noise, hor_m, vert_m):
    batch, width, height = noise.shape
    img_noise = np.zeros([batch, width*hor_m, height*vert_m, 3])
    for l in range(batch):
        for i in range(width):
            for j in range(height):
                img_noise[l, i*hor_m: (i+1)*hor_m, j*vert_m: (j+1)*vert_m] = noise[l, i, j] * 127.5 + 127.5
    return img_noise


if __name__ == '__main__':
    project = "diamonds"
    p_numh, p_numv = 8, 8
    noise_w, noise_h = 16, 16
    img_size = (128, 128)

    generator_path = f'projects\\{project}\\generator'
    generator = tf.keras.models.load_model(generator_path)

    #noise_source = backtesting.full_circle_gradient(8, 8, val_range=(-1, 1))
    #noise = noise_source.reshape((64, 64))
    noise_source = np.random.uniform(-1, 1, (p_numh * p_numv, noise_w, noise_h))
    noise = noise_source.reshape((p_numh * p_numv, noise_w * noise_h))
    #noise_source = backtesting.transition2D(noise_w, noise_h, p_numh, p_numv, val_range=(-1, 1))
    #noise = noise_source.reshape((p_numh * p_numv, noise_w * noise_h))

    gen_imgs = generator.predict(noise, batch_size=8)
    gen_imgs = 127.5 * gen_imgs + 127.5

    prepared_noise = prepare_noise(noise_source, img_size[0] // noise_w, img_size[1] // noise_h)
    stacked_images = np.concatenate([gen_imgs, prepared_noise], axis=2)
    np_image = stack_images(stacked_images, img_size[0], img_size[1] * 2, p_numh, p_numv)
    im_image = Image.fromarray(np_image, 'RGB')
    im_image.save(f"projects\\{project}\\result.png")
import numpy as np


def full_circle_gradient(x_count, y_count, discret=0, val_range=(0, 1)):
    if not discret:
        count = x_count * y_count
    assert discret <= x_count * y_count, f"Max count of gradients is {x_count * y_count}"
    assert x_count >= 2, "x_count should be greater than 1"
    assert y_count >= 2, "y_count should be greater than 1"
    gradients = np.zeros((discret, x_count, y_count), 'float32')

    map = np.zeros((2 * x_count, 2 * y_count))
    for x in range(2 * x_count):
        for y in range(2 * y_count):
            distance = np.sqrt((x-x_count+0.5)**2 + (y-y_count+0.5)**2)
            max_value = np.sqrt((x_count-0.5)**2 + (y_count-0.5)**2)
            min_value = np.sqrt(0.5**2 + 0.5**2)
            map_range = max_value - min_value
            multiplier = -1 / map_range * (val_range[1] - val_range[0])
            map[x, y] = (distance - min_value - map_range) * multiplier + val_range[0]

    counter = 0
    for x_point in range(x_count):
        for y_point in range(y_count):
            x_max = x_point + x_count
            y_max = y_point + y_count
            gradients[counter] = map[x_point: x_max, y_point: y_max]
            if counter == (discret - 1):
                break
            counter += 1
        if counter == discret:
            break
    return gradients


def transition2D(x_count, y_count, x_discret, y_discret, val_range=(0, 1)):
    assert x_count >= 2, "x_count should be greater than 1"
    assert y_count >= 2, "y_count should be greater than 1"
    gradients = np.zeros((x_discret * y_discret, x_count, y_count), 'float32')

    map_start = np.random.uniform(*val_range, (x_count, y_count))
    map_x = np.random.uniform(*val_range, (x_count, y_count))
    map_y = np.random.uniform(*val_range, (x_count, y_count))
    map_x_d = map_x - map_start
    map_y_d = map_y - map_start

    for i_x in range(x_discret):
        mult_x = i_x / (x_discret - 1)
        for i_y in range(y_discret):
            mult_y = i_y / (y_discret - 1)
            map_num = i_x * y_discret + i_y
            new_map = map_start + map_y_d * mult_y + map_x_d * mult_x
            gradients[map_num] = np.clip(new_map, *val_range)
    return gradients
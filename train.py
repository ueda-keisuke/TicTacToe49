from keras.layers import Activation, Add, BatchNormalization, Conv2D, Dense, GlobalAveragePooling2D, Input
from keras.regularizers import l2
import numpy as np
import pickle
from keras.callbacks import LearningRateScheduler
from keras.models import load_model, save_model
from pathlib import Path
from pv_mcts import to_x


def computational_graph():
    WIDTH = 128
    HEIGHT = 16

    def residual_block(input_layer):
        x = BatchNormalization()(input_layer)
        x = Conv2D(WIDTH, 3, padding='same', use_bias=False, kernel_initializer='he_normal',
                   kernel_regularizer=l2(0.0005))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(WIDTH, 3, padding='same', use_bias=False, kernel_initializer='he_normal',
                   kernel_regularizer=l2(0.0005))(x)
        x = BatchNormalization()(x)
        x = Add()([input_layer, x])
        return x

    input_layer = Input(shape=(19, 19, 17))
    x = Conv2D(WIDTH, 3, padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(0.0005))(input_layer)

    for _ in range(HEIGHT):
        x = residual_block(x)

    x = GlobalAveragePooling2D()(x)
    policy_output = Dense(9, kernel_regularizer=l2(0.0005))(x)
    value_output = Dense(1, kernel_regularizer=l2(0.0005), activation='tanh')(x)

    return input_layer, [policy_output, value_output]


def load_data():
    def load_datum(path):
        with path.open(mode='rb') as f:
            return pickle.load(f)

    states, y_policies, y_values = zip(
        *map(load_datum, tuple(sorted(Path('./data').glob('*.pickle')))[-5000:]))

    return map(np.array, (tuple(map(to_x, states)), y_policies, y_values))


def main():
    xs, y_policies, y_values = load_data()

    model_path = sorted(Path('model/candidates').glob('*.h5'))[-1]
    model = load_model(model_path)

    model.compile(loss=['mean_squared_error', 'mean_squared_error'], optimizer='adam')

    def learning_rate_schedule(epoch):
        if epoch < 50:
            return 0.001
        elif epoch < 75:
            return 0.0005
        else:
            return 0.00025

    model.fit(xs, [y_policies, y_values], 100, 100,
              callbacks=[LearningRateScheduler(learning_rate_schedule)])

    save_model(model, model_path.with_name('{:04}.h5'.format(int(model_path.stem) + 1)))


if __name__ == '__main__':
    main()

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


def portfolio_max_yield(weights: np.ndarray, outputs: np.ndarray):
    return -tf.reduce_sum(tf.matmul(weights, tf.transpose(outputs)))


def main():
    np.random.seed(15)
    ret_mat = np.float32(np.random.normal(0, 0.05, size=(500, 10)))
    train, test = ret_mat[:300, :], ret_mat[300:, :]

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(64, input_shape=(ret_mat.shape[1],), activation='linear'))
    #model.add(tf.keras.layers.Softmax())
    model.add(tf.keras.layers.Dense(64, activation='linear'))
    #model.add(tf.keras.layers.Softmax())
    model.add(tf.keras.layers.Dense(ret_mat.shape[1], activation='linear'))
    model.add(tf.keras.layers.Softmax())
    model.compile(loss=portfolio_max_yield, optimizer=tf.keras.optimizers.RMSprop(0.01))
    model.fit(x=train[:-1, :], y=train[1:, :], batch_size=1, epochs=50)

    pred = model.predict(test[:-1, :])
    port_ret = np.sum(pred * test[1:, :], axis=1)
    plt.plot(port_ret.cumsum(), color='k')
    plt.plot(test.cumsum(axis=0))
    plt.show()


if __name__ == '__main__':
    main()
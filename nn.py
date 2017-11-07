from keras import optimizers
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
import numpy as np

from audioclip import AudioClip

def main():
    ac = AudioClip("/mnt/tower_1tb/music/test.wav", start=0, end=3.5)
    ac.region_setup(
            [0.57, 1.04, 2.0, 2.44],
            ['sil', 'i1', 'sil', 'i1','sil'])
    train_x, train_y = ac.batch()
    train_x = train_x[np.newaxis, :, :]
    train_y = train_y[np.newaxis, :, :]

    print(train_x.shape)
    model = buildmodel(train_x.shape[2], train_x.shape[1])
    model.fit(train_x, train_y, epochs=1000, batch_size=1)

def buildmodel(feature_size, sample_count):
    model = Sequential()
    model.add(LSTM(
        20,
        batch_input_shape=(1, sample_count, feature_size),
        return_sequences=True))
    model.add(Dense(18))
    model.add(Dense(4))

    sgd = optimizers.SGD(lr=0.005, momentum=0.01)
    model.compile(loss="mean_squared_error", optimizer=sgd)

    return model

if __name__ == "__main__":
    main()

import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras import backend
from tensorflow.keras.regularizers import l2


def model_small_cnn(input_shape, classes):
    """
    This is the model from "adversarial example detection by classification for
    deep speech recognition" that was published in "ICASSP 2020 - 2020 IEEE
    International Conference on Acoustics, Speech and Signal Processing
    (ICASSP)"
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9054750

    This architecture is also used in Convolutional neural networks for
    small-footprint keyword spotting (by google) and "Convolutional Neural
    Networks for Speech Recognition".
    """
    loss = "sparse_categorical_crossentropy"
    learning_rate = 0.0001

    model = tf.keras.models.Sequential()

    # 1st conv layer
    model.add(layers.Conv2D(64, (2, 2), activation='relu',
              input_shape=input_shape, kernel_regularizer=l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((1, 3), padding='same'))

    # 2nd conv layer
    model.add(layers.Conv2D(64, (2, 2), activation='relu',
              kernel_regularizer=l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2), padding='same'))

    # 3rd conv layer
    model.add(layers.Conv2D(32, (2, 2), activation='relu',
              kernel_regularizer=l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2), padding='same'))
    model.add(layers.Dropout(0.4))

    # flatten output and feed into dense layer
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))

    # Dropout
    model.add(layers.Dropout(0.5))

    # softmax output layer
    model.add(layers.Dense(classes, activation='softmax'))

    # compile model
    optimiser = tf.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimiser, loss=loss, metrics=["accuracy"])

    model.summary()
    return model


def model_large_cnn(input_shape, classes):
    """
    The model used in "Trojaning attacks on neural networks" (NDSS '18)

    In that paper the authors used a (512, 512) spectrogram
    but in our case we will use the MFCCs for consistency with the other
    experiments.
    """
    loss = "sparse_categorical_crossentropy"
    learning_rate = 0.0001

    model = tf.keras.models.Sequential()
    model.add(layers.Conv2D(96, (3, 3), padding="same",
                            input_shape=input_shape,
                            kernel_regularizer=l2(0.001)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(256, (3, 3), padding="same",
                            kernel_regularizer=l2(0.001)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(384, (3, 3), padding="same", activation="relu",
                            kernel_regularizer=l2(0.001)))
    model.add(layers.Conv2D(384, (3, 3), padding="same", activation="relu",
                            kernel_regularizer=l2(0.001)))
    model.add(layers.Conv2D(256, (3, 3), padding="same", activation="relu",
                            kernel_regularizer=l2(0.001)))
    model.add(layers.MaxPooling2D((3, 3), strides=(2, 2)))

    # Use smaller dropout ratios here as it seems that dropout may be the
    # reason behind the inability to train in a few cases. We suspect that
    # large dropout ratios lead to dying ReLUs. However, we need to further
    # investigate this to be 100% sure.
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(classes, activation="softmax"))

    # compile model
    optimiser = tf.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimiser, loss=loss, metrics=["accuracy"])

    model.summary()
    return model


def model_lstm(input_shape, classes):
    """
    An LSTM with attention model.

    This model is the Attention RNN shown in the kaggle competition about the
    speech commands dataset (
    https://paperswithcode.com/sota/keyword-spotting-on-google-speech-commands)

    Its code is published in github
    (https://github.com/douglas125/SpeechCmdRecognition)
    """
    learning_rate = 0.0001
    loss = "sparse_categorical_crossentropy"
    rnn_func = layers.LSTM
    inputs = layers.Input(input_shape, name='input')

    x = layers.Conv2D(10, (5, 1), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(1, (5, 1), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Lambda(lambda q: backend.squeeze(q, -1), name='squeeze_last_dim')(x)

    x = layers.Bidirectional(rnn_func(64, return_sequences=True))(x)  # [b_s, seq_len, vec_dim]
    x = layers.Bidirectional(rnn_func(64, return_sequences=True))(x)  # [b_s, seq_len, vec_dim]

    xFirst = layers.Lambda(lambda q: q[:, -1])(x)  # [b_s, vec_dim]
    query = layers.Dense(128)(xFirst)

    # dot product attention
    attScores = layers.Dot(axes=[1, 2])([query, x])
    attScores = layers.Softmax(name='attSoftmax')(attScores)  # [b_s, seq_len]

    # rescale sequence
    attVector = layers.Dot(axes=[1, 1])([attScores, x])  # [b_s, vec_dim]

    x = layers.Dense(64, activation='relu')(attVector)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(32)(x)

    output = layers.Dense(classes, activation='softmax', name='output')(x)
    model = Model(inputs=[inputs], outputs=[output])

    # compile model
    optimiser = tf.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimiser, loss=loss, metrics=["accuracy"])

    model.summary()
    return model


def get_model(arch, input_shape, classes):
    """Build the model for experiments."""
    if arch == "large_cnn":
        return model_large_cnn(input_shape, classes)
    elif arch == "small_cnn":
        return model_small_cnn(input_shape, classes)
    elif arch == "lstm":
        return model_lstm(input_shape, classes)


if __name__ == "__main__":
    archs = ["small_cnn", "large_cnn", "lstm"]
    # The size of our input features. This is the result for the speech
    # comamnds dataset (1 sec of audio samples at 16kHz) with 40 bins for the
    # MFCCs.
    input_shape = (101, 40, 1)
    classes = 30
    for arch in archs:
        build_model(arch, input_shape, classes)

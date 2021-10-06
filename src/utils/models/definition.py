import tensorflow as tf


def define_models(shape):
    models = [

        (
            'FFNN1',
            [
                tf.keras.layers.Dense(256, activation='relu', input_shape=(shape, )),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(28, activation='softmax'),
            ],

        ),

        (
            'FFNN2',
            [
                tf.keras.layers.Dense(256, activation='relu', input_shape=(shape,)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(28, activation='softmax'),
            ],
        ),


        (
            'FFNN3',
            [
                tf.keras.layers.Dense(512, activation='relu', input_shape=(shape,)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(28, activation='softmax'),
            ],

        ),

        (
            'FFNN4',
            [
                tf.keras.layers.Dense(1024, activation='relu', input_shape=(shape,)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(28, activation='softmax'),
            ],

        ),

        (
            'FFNN6',
            [
                tf.keras.layers.Dense(2048, activation='relu', input_shape=(shape,)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(1024, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(28, activation='softmax'),
            ],

        ),
    ]

    return models

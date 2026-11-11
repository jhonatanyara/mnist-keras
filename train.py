"""MNIST Digit Classification Model Training Script.

This script downloads the MNIST dataset, builds a basic dense neural network
using TensorFlow/Keras, trains the model for 5 epochs, and saves the trained
model to 'modelo_mnist.keras'.
"""

import tensorflow as tf


def main():
    print("Loading MNIST dataset...")
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize pixel values to [0.0, 1.0]
    x_train, x_test = x_train / 255.0, x_test / 255.0

    print("Building Keras Sequential model...")
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    print("Compiling model...")
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    print("Training model for 5 epochs...")
    model.fit(
        x_train,
        y_train,
        epochs=5,
        validation_data=(x_test, y_test)
    )

    output_filename = "modelo_mnist.keras"
    print(f"Saving model to {output_filename}...")
    model.save(output_filename)
    print("Training complete and model successfully saved.")


if __name__ == "__main__":
    main()

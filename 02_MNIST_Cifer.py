import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

print("Training images :", x_train.shape[0])
print("Testing images  :", x_test.shape[0])

plt.figure(figsize=(8,8))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(x_train[i])
    plt.axis("off")
plt.suptitle("Sample CIFAR-10 Images")
plt.show()

x_train = x_train / 255.0
x_test  = x_test / 255.0

y_train_flat = y_train.reshape(-1)
y_test_flat  = y_test.reshape(-1)

ann = models.Sequential([
    layers.Flatten(input_shape=(32, 32, 3)),
    layers.Dense(256, activation="relu"),
    layers.Dense(10, activation="softmax")
])

ann.compile(optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"])

print("\nTraining ANN on CIFAR-10...")
history_ann = ann.fit(x_train, y_train_flat, epochs=10,
                      validation_data=(x_test, y_test_flat))

cnn = models.Sequential([
    layers.Conv2D(32, (3,3), activation="relu", input_shape=(32,32,3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax")
])

cnn.compile(optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"])

print("\nTraining CNN on CIFAR-10...")
history_cnn = cnn.fit(x_train, y_train_flat, epochs=10,
                      validation_data=(x_test, y_test_flat))

# MNIST
(m_train, ml_train), (m_test, ml_test) = datasets.mnist.load_data()

print("Training images :", m_train.shape[0])
print("Testing images  :", m_test.shape[0])

plt.figure(figsize=(8,8))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(m_train[i], cmap="gray")
    plt.axis("off")
plt.suptitle("Sample MNIST Images")
plt.show()

m_train = m_train / 255.0
m_test  = m_test / 255.0

m_train_flat = m_train.reshape((m_train.shape[0], -1))
m_test_flat  = m_test.reshape((m_test.shape[0], -1))

mnist_ann = models.Sequential([
    layers.Dense(128, activation="relu", input_shape=(784,)),
    layers.Dense(10, activation="softmax")
])

mnist_ann.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

print("\nTraining ANN on MNIST...")
hist_mnist_ann = mnist_ann.fit(m_train_flat, ml_train, epochs=10,
                               validation_data=(m_test_flat, ml_test))

m_train_cnn = m_train.reshape(-1, 28, 28, 1)
m_test_cnn  = m_test.reshape(-1, 28, 28, 1)

mnist_cnn = models.Sequential([
    layers.Conv2D(32, (3,3), activation="relu", input_shape=(28,28,1)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax")
])

mnist_cnn.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

print("\nTraining CNN on MNIST...")
hist_mnist_cnn = mnist_cnn.fit(
    m_train_cnn, ml_train, epochs=10,
    validation_data=(m_test_cnn, ml_test)
)


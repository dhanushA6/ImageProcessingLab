import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os

train_dir = r"Horse_Human\train"
test_dir  = r"Horse_Human\test"

train_gen = ImageDataGenerator(
    rescale=1/255.0
)

test_gen = ImageDataGenerator(rescale=1/255.0)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode="categorical"
)

test_data = test_gen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode="categorical"
)

print("Training images:", train_data.samples)
print("Testing images :", test_data.samples)

x, y = next(train_data)
plt.figure(figsize=(8, 8))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(x[i])
    plt.axis("off")
plt.suptitle("Sample Training Images")
plt.show()

def build_resnet():
    base = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(128,128,3)
    )
    base.trainable = False 

    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dense(2, activation="sigmoid")
    ])

    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model

model = build_resnet()
history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=10
)

print("Train Accuracy:", max(history.history['accuracy']))
print("Test Accuracy :", max(history.history['val_accuracy']))

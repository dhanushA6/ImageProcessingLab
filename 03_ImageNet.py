import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

ds, info = tfds.load("imagenet_v2", split="test", as_supervised=True, with_info=True)

NUM_CLASSES = info.features["label"].num_classes
print("ImageNetv2 Classes:", NUM_CLASSES)

SUBSET_SIZE = 2000
small_ds = ds.take(SUBSET_SIZE)

train_size = int(0.7 * SUBSET_SIZE)
test_size  = SUBSET_SIZE - train_size

train_ds = small_ds.take(train_size)
test_ds  = small_ds.skip(train_size)

IMG_SIZE = (128, 128)
BATCH = 32

def preprocess(image, label):
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label

train_data = train_ds.map(preprocess).batch(BATCH).prefetch(1)
test_data  = test_ds.map(preprocess).batch(BATCH).prefetch(1)

print("Training samples:", train_size)
print("Testing samples :", test_size)

sample_batch = next(iter(train_data))
x, y = sample_batch

plt.figure(figsize=(8, 8))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(x[i])
    plt.axis("off")
plt.suptitle("Sample Images from ImageNet-v2 Subset")
plt.show()

augment = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomBrightness(0.2)
])

def augment_fn(image, label):
    image = augment(image)
    return image, label

aug_data = train_data.map(lambda x, y: augment_fn(x, y))

def build_cnn():
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

model1 = build_cnn()
history1 = model1.fit(
    train_data,
    validation_data=test_data,
    epochs=3
)

model2 = build_cnn()
history2 = model2.fit(
    aug_data,
    validation_data=test_data,
    epochs=3
)

print("\n=== Accuracy Comparison ===")
print("Before Augmentation - Train Accuracy:", max(history1.history['accuracy']))
print("Before Augmentation - Test Accuracy :", max(history1.history['val_accuracy']))

print("After Augmentation  - Train Accuracy:", max(history2.history['accuracy']))
print("After Augmentation  - Test Accuracy :", max(history2.history['val_accuracy']))

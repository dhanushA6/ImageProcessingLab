import os
import shutil
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

TRAIN_DIR = r"Fruits Classification\train"
TEST_DIR = r"Fruits Classification\test"

AUG_DIR = TRAIN_DIR + "_AUG"
MERGED_DIR = TRAIN_DIR + "_MERGED"

IMG_SIZE = (128, 128)
BATCH = 32
SAVE_AUG_PER_CLASS = 20 
EPOCHS = 10

gen_orig = ImageDataGenerator(rescale=1 / 255.0)
train_orig = gen_orig.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH, class_mode="categorical"
)
test_orig = gen_orig.flow_from_directory(
    TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH, class_mode="categorical"
)

print("\nOriginal Training images:", train_orig.samples)
print("Original Testing images :", test_orig.samples)

def build_cnn(num_classes):
    model = models.Sequential(
        [
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D(2, 2),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


num_classes = len(train_orig.class_indices)

model_orig = build_cnn(num_classes)
history_orig = model_orig.fit(train_orig, validation_data=test_orig, epochs=EPOCHS)

os.makedirs(AUG_DIR, exist_ok=True)
classes = sorted(
    [d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))]
)

for cls in classes:
    class_input_path = os.path.join(TRAIN_DIR, cls)
    class_aug_path = os.path.join(AUG_DIR, cls)
    os.makedirs(class_aug_path, exist_ok=True)

    gen = ImageDataGenerator(
        rotation_range=25,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.7, 1.3],
    )

    flow = gen.flow_from_directory(
        TRAIN_DIR,
        classes=[cls],
        target_size=IMG_SIZE,
        batch_size=1,
        save_to_dir=class_aug_path,
        save_prefix="aug",
        class_mode=None,
    )

    print(f"Generating {SAVE_AUG_PER_CLASS} images for class: {cls}")
    for _ in range(SAVE_AUG_PER_CLASS):
        next(flow)

print("Done generating augmented images!")

if os.path.exists(MERGED_DIR):
    shutil.rmtree(MERGED_DIR)

shutil.copytree(TRAIN_DIR, MERGED_DIR)

for cls in classes:
    aug_class_path = os.path.join(AUG_DIR, cls)
    merged_class_path = os.path.join(MERGED_DIR, cls)
    for f in os.listdir(aug_class_path):
        shutil.copy(os.path.join(aug_class_path, f), merged_class_path)

print("Merged dataset created at:", MERGED_DIR)

gen_merged = ImageDataGenerator(rescale=1 / 255.0, validation_split=0.2)
train_merged = gen_merged.flow_from_directory(
    MERGED_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode="categorical",
    subset="training",
)
test_merged = gen_merged.flow_from_directory(
    MERGED_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode="categorical",
    subset="validation",
)

model_aug = build_cnn(num_classes)
history_aug = model_aug.fit(train_merged, validation_data=test_merged, epochs=EPOCHS)

print("\n=== Accuracy Comparison ===")
print(f"Original Only - Train Accuracy: {max(history_orig.history['accuracy']):.4f}")
print(
    f"Original Only - Test Accuracy : {max(history_orig.history['val_accuracy']):.4f}"
)
print(f"With Augmentation - Train Accuracy: {max(history_aug.history['accuracy']):.4f}")
print(
    f"With Augmentation - Test Accuracy : {max(history_aug.history['val_accuracy']):.4f}"
)

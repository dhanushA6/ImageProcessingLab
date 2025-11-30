import os
import shutil
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

DATASET_DIR = r"Grapevine_Leaves_Image_Dataset"
IMG_SIZE = (128, 128)
BATCH = 32
EPOCHS = 10
VAL_SPLIT = 0.2
SAVE_AUG_PER_CLASS = 20
AUG_DIR = DATASET_DIR + "_AUG"
MERGED_DIR = DATASET_DIR + "_MERGED"

gen_orig = ImageDataGenerator(rescale=1 / 255.0, validation_split=VAL_SPLIT)

train_orig = gen_orig.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode="categorical",
    subset="training",
    shuffle=True,
)

val_orig = gen_orig.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
)

print("\nTraining images (Original):", train_orig.samples)
print("Validation images (Original):", val_orig.samples)

def build_model(num_classes):
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

model_orig = build_model(num_classes)
history_orig = model_orig.fit(train_orig, epochs=EPOCHS, validation_data=val_orig)

os.makedirs(AUG_DIR, exist_ok=True)
classes = [cls for cls in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, cls))]

for cls in classes:
    cls_dir = os.path.join(DATASET_DIR, cls)
    cls_aug_dir = os.path.join(AUG_DIR, cls)
    os.makedirs(cls_aug_dir, exist_ok=True)

    aug = ImageDataGenerator(
        rotation_range=30,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.7, 1.3],
    )

    flow = aug.flow_from_directory(
        DATASET_DIR,
        classes=[cls],
        target_size=IMG_SIZE,
        batch_size=1,
        save_to_dir=cls_aug_dir,
        save_prefix="aug",
        class_mode=None,
        shuffle=True,
    )

    for _ in range(SAVE_AUG_PER_CLASS):
        next(flow)

print("\nAll augmented images saved!")

if os.path.exists(MERGED_DIR):
    shutil.rmtree(MERGED_DIR)

shutil.copytree(DATASET_DIR, MERGED_DIR)

for cls in classes:
    aug_cls_dir = os.path.join(AUG_DIR, cls)
    merged_cls_dir = os.path.join(MERGED_DIR, cls)
    for f in os.listdir(aug_cls_dir):
        shutil.copy(os.path.join(aug_cls_dir, f), merged_cls_dir)

print("Merged dataset created at:", MERGED_DIR)

gen_merged = ImageDataGenerator(rescale=1 / 255.0, validation_split=VAL_SPLIT)

train_merged = gen_merged.flow_from_directory(
    MERGED_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode="categorical",
    subset="training",
    shuffle=True,
)

val_merged = gen_merged.flow_from_directory(
    MERGED_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
)

model_aug = build_model(num_classes)
history_aug = model_aug.fit(train_merged, epochs=EPOCHS, validation_data=val_merged)

print("\n=== Accuracy Comparison ===")
print(f"Original Only - Train Accuracy: {max(history_orig.history['accuracy']):.4f}")
print(f"Original Only - Val Accuracy  : {max(history_orig.history['val_accuracy']):.4f}")
print(f"With Augmentation - Train Accuracy: {max(history_aug.history['accuracy']):.4f}")
print(f"With Augmentation - Val Accuracy  : {max(history_aug.history['val_accuracy']):.4f}")

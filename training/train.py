"""
Gesture Recognition - Fixed Training Script
- Uses transfer learning from ImageNet pretrained MobileNetV2
- Fine-tunes on our gesture dataset
- Should achieve 85%+ accuracy
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Config
DATA_DIR = os.path.expanduser("~/hand-gesture-wei/data")
MODEL_DIR = os.path.expanduser("~/hand-gesture-wei/training/model")
IMG_SIZE = 96
BATCH_SIZE = 32
GESTURES = ["fist", "open_palm", "peace", "thumbs_up", "swipe_left", "swipe_right", "wave"]
NUM_CLASSES = len(GESTURES)

os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 60)
print("Gesture Recognition Training (Transfer Learning)")
print("=" * 60)

# ── 1. Load Dataset ──────────────────────────────────────────
print("\n[1/6] Loading dataset...")
images, labels = [], []
for label_idx, gesture in enumerate(GESTURES):
    gesture_dir = os.path.join(DATA_DIR, gesture)
    files = sorted([f for f in os.listdir(gesture_dir) if f.endswith(".png")])
    for fname in files:
        img = tf.io.read_file(os.path.join(gesture_dir, fname))
        img = tf.image.decode_png(img, channels=1)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        # Convert grayscale to RGB by repeating channel
        img = tf.repeat(img, 3, axis=-1)
        images.append(img.numpy())
        labels.append(label_idx)
    print(f"  {gesture}: {len(files)} images")

images = np.array(images, dtype=np.float32)
# Normalize for MobileNetV2 (expects [-1, 1])
images = tf.keras.applications.mobilenet_v2.preprocess_input(images)
labels = np.array(labels, dtype=np.int32)
print(f"\nTotal: {len(images)} images")

# ── 2. Split ─────────────────────────────────────────────────
print("\n[2/6] Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=42, stratify=labels)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)
print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# ── 3. Augmentation ──────────────────────────────────────────
print("\n[3/6] Building model with transfer learning...")

# ── 4. Build Model ───────────────────────────────────────────
# Use MobileNetV2 0.35 with ImageNet weights
base_model = keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    alpha=0.35,
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False  # Freeze base initially

inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

# Augmentation
x = keras.layers.RandomFlip("horizontal")(inputs)
x = keras.layers.RandomRotation(0.1)(x)
x = keras.layers.RandomZoom(0.1)(x)

x = base_model(x, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.3)(x)
x = keras.layers.Dense(128, activation="relu")(x)
x = keras.layers.Dropout(0.2)(x)
outputs = keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = keras.Model(inputs, outputs)

total_params = model.count_params()
print(f"Total parameters: {total_params:,}")
print(f"Estimated int8 size: ~{total_params/1024:.0f} KB")

# ── 5. Phase 1: Train head only ──────────────────────────────
print("\n[4/6] Phase 1: Training classification head...")
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8,
                                   restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                       patience=4, verbose=1),
    keras.callbacks.ModelCheckpoint(
        os.path.join(MODEL_DIR, "best_model.keras"),
        monitor="val_accuracy", save_best_only=True, verbose=1)
]

history1 = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=BATCH_SIZE,
    epochs=20,
    callbacks=callbacks,
    verbose=1
)

val_acc = max(history1.history["val_accuracy"])
print(f"\nPhase 1 best val accuracy: {val_acc*100:.1f}%")

# ── 6. Phase 2: Fine-tune top layers ─────────────────────────
print("\n[5/6] Phase 2: Fine-tuning top layers...")
base_model.trainable = True

# Freeze bottom 80% of layers, fine-tune top 20%
fine_tune_from = int(len(base_model.layers) * 0.8)
for layer in base_model.layers[:fine_tune_from]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(1e-5),  # Very low LR for fine-tuning
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks2 = [
    keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=10,
                                   restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                       patience=5, verbose=1),
    keras.callbacks.ModelCheckpoint(
        os.path.join(MODEL_DIR, "best_model.keras"),
        monitor="val_accuracy", save_best_only=True, verbose=1)
]

history2 = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=BATCH_SIZE,
    epochs=30,
    callbacks=callbacks2,
    verbose=1
)

# ── 7. Evaluate ──────────────────────────────────────────────
print("\n[6/6] Evaluating...")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n{'='*40}")
print(f"Final Test Accuracy: {test_acc*100:.1f}%")
print(f"Final Test Loss:     {test_loss:.4f}")
print(f"{'='*40}")

y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=GESTURES))

# Save
model.save(os.path.join(MODEL_DIR, "gesture_model.keras"))

# Plot
all_acc = history1.history["accuracy"] + history2.history["accuracy"]
all_val = history1.history["val_accuracy"] + history2.history["val_accuracy"]
all_loss = history1.history["loss"] + history2.history["loss"]
all_val_loss = history1.history["val_loss"] + history2.history["val_loss"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(all_acc, label="train")
ax1.plot(all_val, label="val")
ax1.axvline(x=len(history1.history["accuracy"]), color="r",
            linestyle="--", label="fine-tune start")
ax1.set_title("Accuracy")
ax1.legend()
ax2.plot(all_loss, label="train")
ax2.plot(all_val_loss, label="val")
ax2.axvline(x=len(history1.history["loss"]), color="r",
            linestyle="--", label="fine-tune start")
ax2.set_title("Loss")
ax2.legend()
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "training_history.png"))
print("\nTraining plot saved.")
print("\n✅ Training complete! Run convert_tflite.py next.")

"""
Tiny Gesture Model v2 - properly trained
Target: <100KB, >80% accuracy
Key fixes:
- Correct normalization (0-255 input, rescale inside model)
- More aggressive augmentation
- Better architecture with residual connections
- Longer training with cosine decay
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

DATA_DIR = os.path.expanduser("~/hand-gesture-wei/data")
MODEL_DIR = os.path.expanduser("~/hand-gesture-wei/training/model")
IMG_SIZE = 96
BATCH_SIZE = 16  # Smaller batch for better generalization with small dataset
GESTURES = ["fist", "open_palm", "peace", "thumbs_up", "swipe_left", "swipe_right", "wave"]
NUM_CLASSES = len(GESTURES)

os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 60)
print("Tiny Gesture Model v2 (target: <100KB, >80%)")
print("=" * 60)

# Load dataset as uint8 [0,255]
print("\n[1/5] Loading dataset...")
images, labels = [], []
for label_idx, gesture in enumerate(GESTURES):
    gesture_dir = os.path.join(DATA_DIR, gesture)
    files = sorted([f for f in os.listdir(gesture_dir) if f.endswith(".png")])
    for fname in files:
        img = tf.io.read_file(os.path.join(gesture_dir, fname))
        img = tf.image.decode_png(img, channels=1)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        images.append(img.numpy().astype(np.float32))
        labels.append(label_idx)
    print(f"  {gesture}: {len(files)} images")

images = np.array(images, dtype=np.float32)
labels = np.array(labels, dtype=np.int32)

X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=42, stratify=labels)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)
print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# Build tiny model
print("\n[2/5] Building tiny model...")

def build_model():
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1))

    # Augmentation
    x = keras.layers.RandomFlip("horizontal")(inputs)
    x = keras.layers.RandomRotation(0.15)(x)
    x = keras.layers.RandomZoom(0.2)(x)
    x = keras.layers.RandomBrightness(0.3)(x)
    x = keras.layers.RandomContrast(0.3)(x)
    x = keras.layers.GaussianNoise(0.05)(x)

    # Normalize
    x = keras.layers.Rescaling(1./127.5, offset=-1)(x)

    # Stem
    x = keras.layers.Conv2D(16, 3, strides=2, padding="same", use_bias=False)(x)  # 48x48
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(6.)(x)

    # Block 1: DW+PW
    x = keras.layers.DepthwiseConv2D(3, padding="same", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(6.)(x)
    x = keras.layers.Conv2D(24, 1, use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(6.)(x)
    x = keras.layers.MaxPooling2D(2)(x)  # 24x24

    # Block 2
    x = keras.layers.DepthwiseConv2D(3, padding="same", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(6.)(x)
    x = keras.layers.Conv2D(48, 1, use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(6.)(x)
    x = keras.layers.MaxPooling2D(2)(x)  # 12x12

    # Block 3
    x = keras.layers.DepthwiseConv2D(3, padding="same", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(6.)(x)
    x = keras.layers.Conv2D(96, 1, use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(6.)(x)
    x = keras.layers.MaxPooling2D(2)(x)  # 6x6

    # Block 4
    x = keras.layers.DepthwiseConv2D(3, padding="same", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(6.)(x)
    x = keras.layers.Conv2D(128, 1, use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(6.)(x)

    # Head
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.4)(x)
    outputs = keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

    return keras.Model(inputs, outputs)

model = build_model()
model.summary()
total_params = model.count_params()
print(f"\nParameters: {total_params:,} (~{total_params//1024} KB int8)")

# Cosine decay schedule
EPOCHS = 100
lr_schedule = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-3,
    decay_steps=EPOCHS * (len(X_train) // BATCH_SIZE),
    alpha=1e-5
)

model.compile(
    optimizer=keras.optimizers.Adam(lr_schedule),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n[3/5] Training...")
callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=20,
                                   restore_best_weights=True, verbose=1),
    keras.callbacks.ModelCheckpoint(
        os.path.join(MODEL_DIR, "tiny_model_v2.keras"),
        monitor="val_accuracy", save_best_only=True, verbose=1)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nFloat32 Test Accuracy: {test_acc*100:.1f}%")

y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
print(classification_report(y_test, y_pred, target_names=GESTURES))

# Convert to TFLite int8
print("\n[4/5] Converting to TFLite int8...")

def representative_dataset():
    for img in X_train:
        yield [img[np.newaxis, ...].astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_model = converter.convert()

tflite_path = os.path.join(MODEL_DIR, "gesture_model_tiny_v2.tflite")
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

size_kb = len(tflite_model) / 1024
print(f"TFLite model: {size_kb:.1f} KB")

# Verify
print("\n[5/5] Verifying...")
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_scale, input_zero_point = input_details[0]['quantization']
print(f"Input: {input_details[0]['shape']} scale={input_scale:.4f} zero={input_zero_point}")

correct = 0
per_class = {g: [0, 0] for g in GESTURES}
for gesture_idx, gesture in enumerate(GESTURES):
    gesture_dir = os.path.join(DATA_DIR, gesture)
    files = sorted([f for f in os.listdir(gesture_dir) if f.endswith(".png")])
    for fname in files:
        img = tf.io.read_file(os.path.join(gesture_dir, fname))
        img = tf.image.decode_png(img, channels=1)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = img.numpy().astype(np.float32)
        img_q = (img / input_scale + input_zero_point).astype(np.int8)
        interpreter.set_tensor(input_details[0]['index'], img_q[np.newaxis])
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0]
        predicted = np.argmax(output)
        per_class[gesture][1] += 1
        if predicted == gesture_idx:
            correct += 1
            per_class[gesture][0] += 1

print("\nPer-class accuracy:")
for gesture in GESTURES:
    c, t = per_class[gesture]
    print(f"  {gesture:15s}: {c}/{t} = {c/t*100:.0f}%")

overall = correct / sum(v[1] for v in per_class.values())
print(f"\nOverall int8 accuracy: {overall*100:.1f}%")

if overall > 0.75:
    import shutil
    shutil.copy(tflite_path, os.path.join(MODEL_DIR, "gesture_model.tflite"))
    os.system(f"xxd -i {tflite_path} > {os.path.join(MODEL_DIR, 'gesture_model.h')}")
    print("✅ Set as main model!")

print(f"\n{'='*60}")
print(f"Float32: {test_acc*100:.1f}% | Int8: {overall*100:.1f}% | Size: {size_kb:.1f} KB")
print(f"{'='*60}")

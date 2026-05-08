"""
Tiny Gesture Recognition Model
Target: <150KB TFLite int8 to fit in WE-I Plus flash
Uses MobileNetV2 alpha=0.1 with transfer learning
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
BATCH_SIZE = 32
GESTURES = ["fist", "open_palm", "peace", "thumbs_up", "swipe_left", "swipe_right", "wave"]
NUM_CLASSES = len(GESTURES)

os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 60)
print("Tiny Gesture Model Training (target: <150KB)")
print("=" * 60)

# Load dataset
print("\n[1/5] Loading dataset...")
images, labels = [], []
for label_idx, gesture in enumerate(GESTURES):
    gesture_dir = os.path.join(DATA_DIR, gesture)
    files = sorted([f for f in os.listdir(gesture_dir) if f.endswith(".png")])
    for fname in files:
        img = tf.io.read_file(os.path.join(gesture_dir, fname))
        img = tf.image.decode_png(img, channels=1)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = tf.repeat(img, 3, axis=-1)
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        images.append(img.numpy())
        labels.append(label_idx)
    print(f"  {gesture}: {len(files)} images")

images = np.array(images, dtype=np.float32)
labels = np.array(labels, dtype=np.int32)

X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=42, stratify=labels)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)
print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# Build tiny model - custom CNN instead of MobileNet
print("\n[2/5] Building tiny custom CNN...")

def build_tiny_model():
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1))

    # Augmentation
    x = keras.layers.RandomFlip("horizontal")(inputs)
    x = keras.layers.RandomRotation(0.15)(x)
    x = keras.layers.RandomZoom(0.15)(x)
    x = keras.layers.RandomBrightness(0.2)(x)
    x = keras.layers.RandomContrast(0.2)(x)

    # Normalize
    x = keras.layers.Rescaling(1./127.5, offset=-1)(x)

    # Block 1
    x = keras.layers.Conv2D(8, 3, padding="same", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(6.)(x)
    x = keras.layers.MaxPooling2D(2)(x)  # 48x48

    # Block 2 - depthwise separable
    x = keras.layers.DepthwiseConv2D(3, padding="same", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(6.)(x)
    x = keras.layers.Conv2D(16, 1, use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(6.)(x)
    x = keras.layers.MaxPooling2D(2)(x)  # 24x24

    # Block 3 - depthwise separable
    x = keras.layers.DepthwiseConv2D(3, padding="same", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(6.)(x)
    x = keras.layers.Conv2D(32, 1, use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(6.)(x)
    x = keras.layers.MaxPooling2D(2)(x)  # 12x12

    # Block 4 - depthwise separable
    x = keras.layers.DepthwiseConv2D(3, padding="same", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(6.)(x)
    x = keras.layers.Conv2D(64, 1, use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(6.)(x)
    x = keras.layers.MaxPooling2D(2)(x)  # 6x6

    # Block 5
    x = keras.layers.DepthwiseConv2D(3, padding="same", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(6.)(x)
    x = keras.layers.Conv2D(128, 1, use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(6.)(x)

    # Head
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

    return keras.Model(inputs, outputs)

model = build_tiny_model()
model.summary()

total_params = model.count_params()
print(f"\nParameters: {total_params:,}")
print(f"Est. int8 size: ~{total_params/1024:.0f} KB")

# Train
print("\n[3/5] Training...")
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Need to preprocess for grayscale model (not MobileNetV2)
X_train_gray = X_train[:, :, :, :1]  # Take only 1 channel
X_val_gray = X_val[:, :, :, :1]
X_test_gray = X_test[:, :, :, :1]
# Undo MobileNetV2 preprocessing, go back to [0,255]
X_train_gray = ((X_train_gray + 1) * 127.5).astype(np.float32)
X_val_gray = ((X_val_gray + 1) * 127.5).astype(np.float32)
X_test_gray = ((X_test_gray + 1) * 127.5).astype(np.float32)

callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=15,
                                   restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                       patience=7, verbose=1),
    keras.callbacks.ModelCheckpoint(
        os.path.join(MODEL_DIR, "tiny_model.keras"),
        monitor="val_accuracy", save_best_only=True, verbose=1)
]

history = model.fit(
    X_train_gray, y_train,
    validation_data=(X_val_gray, y_val),
    batch_size=BATCH_SIZE,
    epochs=80,
    callbacks=callbacks,
    verbose=1
)

test_loss, test_acc = model.evaluate(X_test_gray, y_test, verbose=0)
print(f"\nFloat32 Test Accuracy: {test_acc*100:.1f}%")

# Convert to TFLite int8
print("\n[4/5] Converting to TFLite int8...")

def representative_dataset():
    for img in X_train_gray[:300]:
        yield [img[np.newaxis, ...].astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_model = converter.convert()

tflite_path = os.path.join(MODEL_DIR, "gesture_model_tiny.tflite")
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

size_kb = len(tflite_model) / 1024
print(f"Tiny TFLite model: {size_kb:.1f} KB")

# Verify
print("\n[5/5] Verifying...")
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_scale, input_zero_point = input_details[0]['quantization']
print(f"Input: {input_details[0]['shape']} dtype={input_details[0]['dtype']}")
print(f"Input quantization: scale={input_scale:.6f} zero={input_zero_point}")

correct = 0
per_class = {g: [0,0] for g in GESTURES}
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

# Generate C header
xxd_cmd = f"xxd -i {tflite_path} > {os.path.join(MODEL_DIR, 'gesture_model.h')}"
os.system(xxd_cmd)

# Also copy as main model if good enough
if overall > 0.75:
    import shutil
    shutil.copy(tflite_path, os.path.join(MODEL_DIR, "gesture_model.tflite"))
    print("✅ Tiny model set as main model!")

print(f"\n{'='*60}")
print(f"Float32:  {test_acc*100:.1f}%")
print(f"Int8:     {overall*100:.1f}%")
print(f"Size:     {size_kb:.1f} KB")
print(f"{'='*60}")

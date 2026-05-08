"""
Quantization-Aware Training (QAT)
Finetunes the trained model with fake quantization nodes
to minimize accuracy loss after int8 conversion
"""
import os
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot

DATA_DIR = os.path.expanduser("~/hand-gesture-wei/data")
MODEL_DIR = os.path.expanduser("~/hand-gesture-wei/training/model")
GESTURES = ["fist", "open_palm", "peace", "thumbs_up", "swipe_left", "swipe_right", "wave"]
NUM_CLASSES = len(GESTURES)
IMG_SIZE = 96
BATCH_SIZE = 32

print("=" * 60)
print("Quantization-Aware Training (QAT)")
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

from sklearn.model_selection import train_test_split
images = np.array(images, dtype=np.float32)
labels = np.array(labels, dtype=np.int32)
X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=42, stratify=labels)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)
print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# Load trained model
print("\n[2/5] Loading trained model...")
model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "best_model.keras"))
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Float32 baseline: {test_acc*100:.1f}%")

# Apply QAT
print("\n[3/5] Applying quantization-aware training...")
qat_model = tfmot.quantization.keras.quantize_model(model)
qat_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
qat_model.summary()

# Fine-tune with QAT
print("\n[4/5] Fine-tuning with QAT...")
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=8,
        restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(
        os.path.join(MODEL_DIR, "qat_model.keras"),
        monitor="val_accuracy", save_best_only=True, verbose=1)
]

history = qat_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=BATCH_SIZE,
    epochs=20,
    callbacks=callbacks,
    verbose=1
)

test_loss, test_acc = qat_model.evaluate(X_test, y_test, verbose=0)
print(f"\nQAT model accuracy: {test_acc*100:.1f}%")

# Convert QAT model to TFLite int8
print("\n[5/5] Converting QAT model to TFLite int8...")

def representative_dataset():
    for img in X_train[:200]:
        yield [img[np.newaxis, ...].astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

tflite_path = os.path.join(MODEL_DIR, "gesture_model_qat.tflite")
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

size_kb = len(tflite_model) / 1024
print(f"QAT TFLite model: {size_kb:.1f} KB")

# Verify
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_scale, input_zero_point = input_details[0]['quantization']

correct = 0
for i, (img, label) in enumerate(zip(X_test, y_test)):
    img_q = (img / input_scale + input_zero_point).astype(np.int8)
    interpreter.set_tensor(input_details[0]['index'], img_q[np.newaxis])
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    if np.argmax(output) == label:
        correct += 1

qat_tflite_acc = correct / len(y_test)
print(f"QAT TFLite int8 accuracy: {qat_tflite_acc*100:.1f}%")

# If QAT model is better, use it as the main model
if qat_tflite_acc > 0.838:
    import shutil
    shutil.copy(tflite_path, os.path.join(MODEL_DIR, "gesture_model.tflite"))
    print("✅ QAT model is better — replacing main model!")

    # Regenerate C header
    c_array = ", ".join([f"0x{b:02x}" for b in tflite_model])
    header = f"""// Auto-generated gesture model (QAT) - {size_kb:.1f} KB
// Classes: {', '.join(GESTURES)}
// Accuracy: {qat_tflite_acc*100:.1f}% (int8 QAT)

#ifndef GESTURE_MODEL_H_
#define GESTURE_MODEL_H_

const char* kGestureLabels[] = {{"{('", "'.join(GESTURES))}"}};
const int kNumGestures = {NUM_CLASSES};
const unsigned int gesture_model_len = {len(tflite_model)};
alignas(8) const unsigned char gesture_model_data[] = {{
  {c_array}
}};

#endif  // GESTURE_MODEL_H_
"""
    with open(os.path.join(MODEL_DIR, "gesture_model.h"), "w") as f:
        f.write(header)
    print("C header updated.")

print(f"\n{'='*60}")
print(f"Float32:     {test_acc*100:.1f}%")
print(f"Int8 (QAT):  {qat_tflite_acc*100:.1f}%")
print(f"Model size:  {size_kb:.1f} KB")
print(f"{'='*60}")

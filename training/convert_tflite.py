"""
Convert trained Keras model to TFLite int8 - improved version
Uses better calibration to minimize accuracy drop
"""
import os
import numpy as np
import tensorflow as tf

DATA_DIR = os.path.expanduser("~/hand-gesture-wei/data")
MODEL_DIR = os.path.expanduser("~/hand-gesture-wei/training/model")
GESTURES = ["fist", "open_palm", "peace", "thumbs_up", "swipe_left", "swipe_right", "wave"]
IMG_SIZE = 96

print("=" * 60)
print("TFLite Conversion (int8 quantization)")
print("=" * 60)

# Load model
print("\n[1/4] Loading model...")
model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "best_model.keras"))

# Load ALL images for calibration (more = better quantization)
print("\n[2/4] Loading full calibration dataset...")
cal_images = []
for gesture in GESTURES:
    gesture_dir = os.path.join(DATA_DIR, gesture)
    files = sorted([f for f in os.listdir(gesture_dir) if f.endswith(".png")])
    for fname in files:  # Use ALL images for calibration
        img = tf.io.read_file(os.path.join(gesture_dir, fname))
        img = tf.image.decode_png(img, channels=1)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = tf.repeat(img, 3, axis=-1)
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        cal_images.append(img.numpy())

cal_images = np.array(cal_images, dtype=np.float32)
print(f"Calibration set: {len(cal_images)} images")

def representative_dataset():
    for img in cal_images:
        yield [img[np.newaxis, ...].astype(np.float32)]

# Convert with full integer quantization
print("\n[3/4] Converting to TFLite int8...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

tflite_path = os.path.join(MODEL_DIR, "gesture_model.tflite")
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

size_kb = len(tflite_model) / 1024
print(f"Model size: {size_kb:.1f} KB")

# Verify accuracy on ALL test images
print("\n[4/4] Verifying accuracy on full dataset...")
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_scale, input_zero_point = input_details[0]['quantization']

print(f"Input: {input_details[0]['shape']} dtype={input_details[0]['dtype']}")
print(f"Input quantization: scale={input_scale:.6f}, zero_point={input_zero_point}")

correct = 0
total = 0
per_class = {g: [0, 0] for g in GESTURES}

for gesture_idx, gesture in enumerate(GESTURES):
    gesture_dir = os.path.join(DATA_DIR, gesture)
    files = sorted([f for f in os.listdir(gesture_dir) if f.endswith(".png")])
    for fname in files:
        img = tf.io.read_file(os.path.join(gesture_dir, fname))
        img = tf.image.decode_png(img, channels=1)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = tf.repeat(img, 3, axis=-1)
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img.numpy())
        img_q = (img / input_scale + input_zero_point).astype(np.int8)
        interpreter.set_tensor(input_details[0]['index'], img_q[np.newaxis])
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0]
        predicted = np.argmax(output)
        per_class[gesture][1] += 1
        if predicted == gesture_idx:
            correct += 1
            per_class[gesture][0] += 1
        total += 1

print(f"\nPer-class accuracy:")
for gesture in GESTURES:
    c, t = per_class[gesture]
    print(f"  {gesture:15s}: {c}/{t} = {c/t*100:.0f}%")
print(f"\nOverall TFLite int8 accuracy: {correct}/{total} = {correct/total*100:.1f}%")

# Generate C header
print("\nGenerating C header...")
c_array = ", ".join([f"0x{b:02x}" for b in tflite_model])
header = f"""// Auto-generated gesture model - {size_kb:.1f} KB
// Classes: {', '.join(GESTURES)}
// Accuracy: {correct/total*100:.1f}% (int8)

#ifndef GESTURE_MODEL_H_
#define GESTURE_MODEL_H_

const char* kGestureLabels[] = {{"{'", "'.join(GESTURES)}"}};
const int kNumGestures = 7;
const unsigned int gesture_model_len = {len(tflite_model)};
alignas(8) const unsigned char gesture_model_data[] = {{
  {c_array}
}};

#endif  // GESTURE_MODEL_H_
""".replace("7", str(len(GESTURES)))

with open(os.path.join(MODEL_DIR, "gesture_model.h"), "w") as f:
    f.write(header)

print(f"\n{'='*60}")
print(f"✅ Done! Model: {size_kb:.1f} KB | Accuracy: {correct/total*100:.1f}%")
print(f"{'='*60}")

# Hand Gesture Object Detection

Deploying YOLOv-Fastest on Himax WE-I Plus (HX6537-A, ARC EM9D)

## Architecture
YOLOv-Fastest v1 re-implemented in Keras:
- Backbone: MobileNetV1-based depthwise separable convolutions
- Head: YOLO detection head (2 scales)
- Activations: ReLU6 (replaces LeakyReLU for TFLite Micro compatibility)
- Input: 96x96x1 grayscale
- Output: 2 detection heads (no NMS - implemented in C++)

## Pipeline
1. Collect raw images via SPI streaming
2. Annotate with bounding boxes (LabelImg)
3. Train in Keras with ReLU6 activations
4. Convert to TFLite int8
5. Write C++ YOLO decoder + NMS on board
6. Deploy via himax_tflm build system

## Classes
- fist, open_palm, peace, thumbs_up, swipe_left, swipe_right, wave

## Target Metrics
- Model size: < 500KB int8
- Inference: < 200ms per frame
- mAP: > 60%

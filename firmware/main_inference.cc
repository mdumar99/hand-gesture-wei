#include <stdint.h>
#include <string.h>
#include "capture.h"
#include "hx_drv_tflm.h"
#include "gesture_model.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;
  constexpr int kTensorArenaSize = 220 * 1024;
#if (defined(__GNUC__) || defined(__GNUG__)) && !defined(__CCAC__)
  static uint8_t tensor_arena[kTensorArenaSize] __attribute__((section(".tensor_arena")));
#else
#pragma Bss(".tensor_arena")
  static uint8_t tensor_arena[kTensorArenaSize];
#pragma Bss()
#endif
}

static const char* gesture_labels[] = {
  "fist","open_palm","peace","thumbs_up","swipe_left","swipe_right","wave"
};
static const int kNumGestures = 7;
static int8_t image_buffer[96 * 96];

void setup() {
  tflite::InitializeTarget();
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;
  model = tflite::GetModel(gesture_model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    hx_drv_uart_print("ERROR: schema mismatch\n");
    return;
  }
  static tflite::MicroMutableOpResolver<10> micro_op_resolver;
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddDepthwiseConv2D();
  micro_op_resolver.AddMaxPool2D();
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddSoftmax();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddMean();
  micro_op_resolver.AddAdd();
  micro_op_resolver.AddQuantize();
  micro_op_resolver.AddDequantize();
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    hx_drv_uart_print("ERROR: AllocateTensors failed\n");
    return;
  }
  hx_drv_uart_print("Arena used: %d\n", interpreter->arena_used_bytes());
  input  = interpreter->input(0);
  output = interpreter->output(0);
}

void loop() {
  hx_drv_uart_print("Capturing...\n");
  if (!CaptureFrame(image_buffer)) {
    hx_drv_uart_print("ERROR: Capture failed\n");
    return;
  }
  int8_t* inp = input->data.int8;
  for (int i = 0; i < 96*96; i++) {
    inp[i] = (int8_t)((int)image_buffer[i] - 128);
  }
  if (interpreter->Invoke() != kTfLiteOk) {
    hx_drv_uart_print("ERROR: Invoke failed\n");
    return;
  }
  int8_t* scores = output->data.int8;
  int best_idx = 0;
  int8_t best_score = scores[0];
  for (int i = 1; i < kNumGestures; i++) {
    if (scores[i] > best_score) {
      best_score = scores[i];
      best_idx = i;
    }
  }
  float conf = (best_score - output->params.zero_point) *
                output->params.scale * 100.0f;
  static int frame_count = 0;
  frame_count++;
  hx_drv_uart_print("[%d] %-12s %.0f%%\n", frame_count, gesture_labels[best_idx], conf);
  if (conf > 70.0f) {
    hx_drv_led_on(HX_DRV_LED_GREEN);
    hx_drv_led_off(HX_DRV_LED_RED);
  } else {
    hx_drv_led_off(HX_DRV_LED_GREEN);
    hx_drv_led_on(HX_DRV_LED_RED);
  }
}

int main() {
  hx_drv_uart_initial(UART_BR_115200);
  hx_drv_uart_print("Gesture Recognition v1.0\n");
  setup();
  hx_drv_uart_print("READY\n");
  while (1) { loop(); }
  return 0;
}

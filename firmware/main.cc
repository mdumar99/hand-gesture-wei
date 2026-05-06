#include <stdint.h>
#include <string.h>
#include "capture.h"
#include "hx_drv_tflm.h"

static int8_t image_buffer[IMG_SIZE];

static int uart_readline(char* buf, int maxlen) {
  int i = 0;
  uint8_t c = 0;
  while (i < maxlen - 1) {
    if (hx_drv_uart_getchar(&c) != HX_DRV_LIB_PASS) continue;
    if (c == 0x0A || c == 0x0D) {
      if (i > 0) break;
      continue;
    }
    if (c < 32 || c > 126) continue;
    buf[i++] = (char)c;
  }
  buf[i] = 0;
  return i;
}

int main() {
  hx_drv_uart_initial(UART_BR_115200);
  hx_drv_uart_print("BOOTING\n");

  hx_drv_uart_print("INIT_CAM\n");
  if (!InitCamera()) {
    hx_drv_uart_print("ERROR: Camera init failed\n");
    while (1) {}
  }
  hx_drv_uart_print("CAPTURE_READY\n");

  char cmd[32];

  while (1) {
    uart_readline(cmd, sizeof(cmd));
    if (cmd[0] == 0) continue;

    if (__builtin_strcmp(cmd, "CAPTURE") == 0) {
      if (!CaptureFrame(image_buffer)) {
        hx_drv_uart_print("ERROR: Capture failed\n");
        continue;
      }
      hx_drv_uart_print("IMG_START\n");
      for (int i = 0; i < IMG_SIZE; i++) {
        hx_drv_uart_print("%c", (uint8_t)image_buffer[i]);
      }
      hx_drv_uart_print("IMG_END\n");

    } else if (__builtin_strcmp(cmd, "PING") == 0) {
      hx_drv_uart_print("PONG\n");

    } else {
      hx_drv_uart_print("UNKNOWN_CMD\n");
    }
  }
  return 0;
}

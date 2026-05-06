#include <stdint.h>
#include <string.h>
#include "capture.h"
#include "hx_drv_tflm.h"

static int8_t image_buffer[IMG_SIZE];

static int uart_readline(char* buf, int maxlen) {
  int i = 0;
  while (i < maxlen - 1) {
    char c = 0;
    hx_drv_uart_getchar((uint8_t*)&c);
    if (c == '\n' || c == '\r') break;
    buf[i++] = c;
  }
  buf[i] = '\0';
  return i;
}

int main() {
  hx_drv_uart_initial(UART_BR_921600);

  if (!InitCamera()) {
    hx_drv_uart_print("ERROR: Camera init failed\n");
    while (1) {}
  }

  hx_drv_uart_print("CAPTURE_READY\n");

  char cmd[32];

  while (1) {
    uart_readline(cmd, sizeof(cmd));

    if (strcmp(cmd, "CAPTURE") == 0) {
      if (!CaptureFrame(image_buffer)) {
        hx_drv_uart_print("ERROR: Capture failed\n");
        continue;
      }
      hx_drv_uart_print("IMG_START\n");
      for (int i = 0; i < IMG_SIZE; i++) {
        hx_drv_uart_print("%c", (uint8_t)image_buffer[i]);
      }
      hx_drv_uart_print("IMG_END\n");

    } else if (strcmp(cmd, "PING") == 0) {
      hx_drv_uart_print("PONG\n");

    } else {
      hx_drv_uart_print("UNKNOWN_CMD\n");
    }
  }

  return 0;
}

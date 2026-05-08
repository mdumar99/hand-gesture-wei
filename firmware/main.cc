#include <stdint.h>
#include <string.h>
#include "capture.h"
#include "hx_drv_tflm.h"

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

  // Initialize SPI for image streaming
  if (hx_drv_spim_init() != HX_DRV_LIB_PASS) {
    hx_drv_uart_print("ERROR: SPI init failed\n");
    while (1) {}
  }

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
      uint8_t* jpeg_buf = 0;
      uint32_t jpeg_size = 0;

      if (!CaptureJpeg(&jpeg_buf, &jpeg_size)) {
        hx_drv_uart_print("ERROR: Capture failed\n");
        continue;
      }

      // Send size over UART, then image over SPI
      hx_drv_uart_print("IMG_START %lu\n", jpeg_size);
      hx_drv_spim_send((uint32_t)jpeg_buf, jpeg_size, SPI_TYPE_JPG);
      hx_drv_uart_print("IMG_END\n");

    } else if (__builtin_strcmp(cmd, "PING") == 0) {
      hx_drv_uart_print("PONG\n");

    } else {
      hx_drv_uart_print("UNKNOWN_CMD\n");
    }
  }
  return 0;
}

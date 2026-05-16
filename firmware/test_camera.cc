#include "capture.h"
#include "hx_drv_tflm.h"

int main() {
  hx_drv_uart_initial(UART_BR_115200);
  hx_drv_uart_print("Testing camera...\n");
  
  if (!InitCamera()) {
    hx_drv_uart_print("FAILED\n");
    while(1){}
  }
  hx_drv_uart_print("Camera OK!\n");
  while(1){}
  return 0;
}

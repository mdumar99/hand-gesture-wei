#include "capture.h"
#include "hx_drv_tflm.h"

static hx_drv_sensor_image_config_t g_pimg_config;

bool InitCamera() {
  if (hx_drv_sensor_initial(&g_pimg_config) != HX_DRV_LIB_PASS) {
    return false;
  }
  return true;
}

bool CaptureFrame(int8_t* image_buffer) {
  if (hx_drv_sensor_capture(&g_pimg_config) != HX_DRV_LIB_PASS) {
    return false;
  }
  hx_drv_image_rescale((uint8_t*)g_pimg_config.raw_address,
                        g_pimg_config.img_width,
                        g_pimg_config.img_height,
                        image_buffer,
                        IMG_WIDTH,
                        IMG_HEIGHT);
  return true;
}

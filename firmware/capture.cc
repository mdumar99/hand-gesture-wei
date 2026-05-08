#include "capture.h"
#include "hx_drv_tflm.h"

static hx_drv_sensor_image_config_t g_pimg_config;
static bool is_initialized = false;

bool InitCamera() {
  if (is_initialized) return true;

  g_pimg_config.sensor_type = HX_DRV_SENSOR_TYPE_HM0360_MONO;
  g_pimg_config.format      = HX_DRV_VIDEO_FORMAT_YUV400;
  g_pimg_config.img_width   = 640;
  g_pimg_config.img_height  = 480;

  if (hx_drv_sensor_initial(&g_pimg_config) != HX_DRV_LIB_PASS) {
    return false;
  }
  is_initialized = true;
  return true;
}

bool CaptureJpeg(uint8_t** jpeg_buf, uint32_t* jpeg_size) {
  if (!is_initialized) return false;

  if (hx_drv_sensor_capture(&g_pimg_config) != HX_DRV_LIB_PASS) {
    return false;
  }

  *jpeg_buf  = (uint8_t*)g_pimg_config.jpeg_address;
  *jpeg_size = g_pimg_config.jpeg_size;
  return true;
}

bool CaptureFrame(int8_t* image_buffer) {
  if (!is_initialized) return false;

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

#ifndef CAPTURE_H_
#define CAPTURE_H_

#include <stdint.h>

// Image dimensions
#define IMG_WIDTH  96
#define IMG_HEIGHT 96
#define IMG_SIZE   (IMG_WIDTH * IMG_HEIGHT)

// Initialize camera
bool InitCamera();

// Capture one frame into buffer (int8_t as required by hx_drv_image_rescale)
bool CaptureFrame(int8_t* image_buffer);

#endif  // CAPTURE_H_

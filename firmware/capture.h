#ifndef CAPTURE_H_
#define CAPTURE_H_

#include <stdint.h>

#define IMG_WIDTH  96
#define IMG_HEIGHT 96
#define IMG_SIZE   (IMG_WIDTH * IMG_HEIGHT)

bool InitCamera();
bool CaptureJpeg(uint8_t** jpeg_buf, uint32_t* jpeg_size);
bool CaptureFrame(int8_t* image_buffer);

#endif  // CAPTURE_H_

"""
Real-time gesture data collection for Himax WE-I Plus
- Sends CAPTURE commands over UART
- Receives JPEG images over SPI (via WEI_SPIrecvImg)
- Displays live feed in OpenCV
- Saves 96x96 grayscale images for training
"""
import serial
import subprocess
import numpy as np
import cv2
import os
import time
import glob
import threading

# Config
UART_PORT = "/dev/ttyUSB0"
BAUD = 115200
SPI_TOOL = os.path.expanduser("~/hand-gesture-wei/himax_sdk/SPI_Tool/WEI_SPIrecvImg")
SPI_DIR = os.path.expanduser("~/hand-gesture-wei/himax_sdk/SPI_Tool")
DATA_DIR = os.path.expanduser("~/hand-gesture-wei/data")
GESTURES = ["fist", "open_palm", "peace", "thumbs_up", "swipe_left", "swipe_right", "wave"]


def wait_for_ready(ser):
    print("Waiting for board... press RESET if needed")
    while True:
        line = ser.readline().decode("utf-8", errors="replace").strip()
        if line:
            print(f"  Board: {line}")
        if "CAPTURE_READY" in line:
            print("Board ready!")
            return


def capture_frame(ser):
    """Send CAPTURE command and receive JPEG via SPI tool."""
    # Clean up old dat files
    for f in glob.glob(os.path.join(SPI_DIR, "default*.dat")):
        os.remove(f)

    # Start SPI receiver for 1 image
    spi_proc = subprocess.Popen(
        [SPI_TOOL, "1"],
        cwd=SPI_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    # Send CAPTURE command
    ser.reset_input_buffer()
    ser.write(b"CAPTURE\n")
    ser.flush()

    # Wait for IMG_END on UART
    deadline = time.time() + 5.0
    while time.time() < deadline:
        line = ser.readline().decode("utf-8", errors="replace").strip()
        if "IMG_END" in line:
            break
        if "ERROR" in line:
            spi_proc.terminate()
            return None

    # Wait for SPI tool to finish
    try:
        spi_proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        spi_proc.terminate()
        return None

    # Find the dat file
    dat_files = glob.glob(os.path.join(SPI_DIR, "default*.dat"))
    if not dat_files:
        return None

    # Read JPEG from dat file
    with open(dat_files[0], "rb") as f:
        jpeg_data = f.read()

    if len(jpeg_data) < 100:
        return None

    # Decode JPEG
    arr = np.frombuffer(jpeg_data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    return img


def main():
    # Fix USB permissions
    os.system("sudo chmod 666 /dev/bus/usb/001/* 2>/dev/null")
    os.system(f"sudo chmod 666 {UART_PORT} 2>/dev/null")

    ser = serial.Serial(UART_PORT, BAUD, timeout=5)
    wait_for_ready(ser)

    gesture_idx = 0
    gesture = GESTURES[gesture_idx]
    save_dir = os.path.join(DATA_DIR, gesture)
    count = len([f for f in os.listdir(save_dir) if f.endswith(".png")])

    print(f"\nGesture: {gesture} | Captured: {count}")
    print("SPACE=capture | N=next gesture | D=delete last | Q=quit\n")

    last_frame = None

    while True:
        # Capture frame
        frame = capture_frame(ser)
        if frame is None:
            print("Frame capture failed, retrying...")
            continue

        last_frame = frame

        # Display
        display = cv2.resize(frame, (480, 360))
        display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)

        # Top bar
        cv2.rectangle(display, (0, 0), (480, 40), (30, 30, 30), -1)
        cv2.putText(display, f"Gesture: {gesture}  [{gesture_idx+1}/{len(GESTURES)}]",
                    (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Bottom bar
        cv2.rectangle(display, (0, 320), (480, 360), (30, 30, 30), -1)
        cv2.putText(display, f"Count: {count} | SPACE=save  N=next  D=del  Q=quit",
                    (8, 348), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)

        cv2.imshow("Gesture Capture", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord(" "):
            fname = os.path.join(save_dir, f"{gesture}_{count:04d}.png")
            small = cv2.resize(last_frame, (96, 96))
            cv2.imwrite(fname, small)
            count += 1
            print(f"Saved: {fname} ({count} total)")

        elif key == ord("n"):
            gesture_idx = (gesture_idx + 1) % len(GESTURES)
            gesture = GESTURES[gesture_idx]
            save_dir = os.path.join(DATA_DIR, gesture)
            count = len([f for f in os.listdir(save_dir) if f.endswith(".png")])
            print(f"Switched to: {gesture} | Existing: {count}")

        elif key == ord("d"):
            files = sorted([f for f in os.listdir(save_dir) if f.endswith(".png")])
            if files:
                os.remove(os.path.join(save_dir, files[-1]))
                count -= 1
                print(f"Deleted last. Count: {count}")

    ser.close()
    cv2.destroyAllWindows()
    print("Done!")


if __name__ == "__main__":
    main()

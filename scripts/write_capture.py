script = '''
import serial
import numpy as np
import cv2
import os
import time

PORT = "/dev/ttyUSB0"
BAUD = 115200
IMG_W, IMG_H = 96, 96
IMG_SIZE = IMG_W * IMG_H

GESTURES = ["fist","open_palm","peace","thumbs_up","swipe_left","swipe_right","wave"]
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data")

def wait_for_ready(ser):
    print("Waiting for board... press RESET if needed")
    while True:
        line = ser.readline().decode("utf-8", errors="replace").strip()
        if "CAPTURE_READY" in line:
            print("Board ready!")
            return

def capture_frame(ser):
    ser.reset_input_buffer()
    ser.write(b"CAPTURE\\n")
    ser.flush()
    # Wait for IMG_START
    while True:
        line = ser.readline().decode("utf-8", errors="replace").strip()
        if "IMG_START" in line:
            break
        if "ERROR" in line:
            print("Error:", line)
            return None
    # Read exactly IMG_SIZE bytes
    raw = b""
    while len(raw) < IMG_SIZE:
        chunk = ser.read(IMG_SIZE - len(raw))
        if chunk:
            raw += chunk
    # Read IMG_END
    ser.readline()
    arr = np.frombuffer(raw, dtype=np.uint8).reshape((IMG_H, IMG_W))
    return arr

def main():
    ser = serial.Serial(PORT, BAUD, timeout=5)
    wait_for_ready(ser)

    gesture_idx = 0
    gesture = GESTURES[gesture_idx]
    count = len(os.listdir(os.path.join(DATA_DIR, gesture)))

    print(f"\\nGesture: {gesture} | Captured: {count}")
    print("SPACE=capture | N=next gesture | D=delete last | Q=quit")

    last_frame = None

    while True:
        # Capture a preview frame
        frame = capture_frame(ser)
        if frame is None:
            continue
        last_frame = frame

        # Display with UI overlay
        display = cv2.resize(frame, (384, 384), interpolation=cv2.INTER_NEAREST)
        display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)

        # Draw UI
        cv2.rectangle(display, (0,0), (384,40), (40,40,40), -1)
        cv2.putText(display, f"Gesture: {gesture}", (8,28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.rectangle(display, (0,344), (384,384), (40,40,40), -1)
        cv2.putText(display, f"Count: {count} | SPC=save N=next D=del Q=quit",
                    (8,372), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)

        cv2.imshow("Gesture Capture", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord(" "):
            # Save image
            save_dir = os.path.join(DATA_DIR, gesture)
            fname = os.path.join(save_dir, f"{gesture}_{count:04d}.png")
            cv2.imwrite(fname, frame)
            count += 1
            print(f"Saved: {fname} ({count} total)")

        elif key == ord("n"):
            gesture_idx = (gesture_idx + 1) % len(GESTURES)
            gesture = GESTURES[gesture_idx]
            count = len(os.listdir(os.path.join(DATA_DIR, gesture)))
            print(f"\\nSwitched to: {gesture} | Existing: {count}")

        elif key == ord("d"):
            save_dir = os.path.join(DATA_DIR, gesture)
            files = sorted(os.listdir(save_dir))
            if files:
                os.remove(os.path.join(save_dir, files[-1]))
                count -= 1
                print(f"Deleted last image. Count: {count}")

    ser.close()
    cv2.destroyAllWindows()
    print("Done!")

if __name__ == "__main__":
    main()
'''
with open("capture_data.py", "w") as f:
    f.write(script)
print("capture_data.py written!")

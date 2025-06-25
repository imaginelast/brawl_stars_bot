from ultralytics import YOLO
import cv2, numpy as np, mss, pygetwindow as gw
import time
import pyautogui

model = YOLO("runs/detect/train/weights/best.engine")

win = gw.getWindowsWithTitle("Bluestacks")[0]
monitor = {"top": win.top, "left": win.left, "width": win.width, "height": win.height}

prev_time = time.time()

clicked_x, clicked_y = -1, -1

def mouse_callback(event, x, y, flags, param):
    global clicked_x, clicked_y
    if event == cv2.EVENT_MOUSEMOVE:
        clicked_x, clicked_y = x, y

cv2.namedWindow("TensorRT Detection")
cv2.setMouseCallback("TensorRT Detection", mouse_callback)

with mss.mss() as sct:
    while True:
        # Measure time
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        # Capture frame
        img = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # Run detection
        results = model.predict(frame, conf=0.4, verbose=False)[0]

        if 0 <= clicked_x < frame.shape[1] and 0 <= clicked_y < frame.shape[0]:
            b, g, r = frame[clicked_y, clicked_x]
            print(f"Mouse at ({clicked_x}, {clicked_y}) - Color: (BGR) ({b}, {g}, {r})")

        # Draw boxes
        for box in results.boxes:
            cls_id = int(box.cls[0])
            x, y, w, h = box.xywh[0]
            x, y, w, h = int(x), int(y), int(w), int(h)
            cv2.rectangle(frame, (x - w//2, y - h//2), (x + w//2, y + h//2), (0,255,0), 2)
            label = results.names[cls_id]
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        

        # Draw FPS
        cv2.putText(frame, f"{fps:.1f} FPS", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

        # Show result
        cv2.imshow("TensorRT Detection", frame)


        if cv2.waitKey(1) == 27:
            break

cv2.destroyAllWindows()

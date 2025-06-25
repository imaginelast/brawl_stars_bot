import cv2
import numpy as np
from ultralytics import YOLO
import mss
import pygetwindow as gw
import pyautogui
import time

#model using Tensor RT
model = YOLO("runs/detect/train/weights/best.engine") 

win = gw.getWindowsWithTitle("Bluestacks")[0]
monitor = {
    "top": win.top,
    "left": win.left,
    "width": win.width,
    "height": win.height
}

#ellipse around player (used for detecting if enemy is in player range)
def is_in_ellipse(ex, ey, px, py, a, b):
    return ((ex - px) ** 2) / a ** 2 + ((ey - py) ** 2) / b ** 2 <= 1


#var
h_ellipse = 240  
v_ellipse = 200 
prev_time = time.time()
last_move_time = time.time()

held_keys = {"w": False, "a": False, "s": False, "d": False}

mouse_x, mouse_y = pyautogui.position()


# (chatgpt) functions used to move
def hold_key(key):
    if not held_keys[key]:
        pyautogui.keyDown(key)
        held_keys[key] = True

def release_key(key):
    if held_keys[key]:
        pyautogui.keyUp(key)
        held_keys[key] = False

def release_all_keys():
    for key in held_keys:
        release_key(key)

start = True

with mss.mss() as sct:
    while True:

        #getting frame of bluestacks window
        screen = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)


        #run detection 
        results = model.predict(frame, conf=0.4, verbose=False)[0] #getting all objects in frame with confidence >= 0.4

        annotated_frame = results.plot() #frame with objects shown in frame

        player_center = None
        enemy_centers = []

        #super color 1454 745 rgb: 255 193 30
        # b1, g1, r1 = frame[745 - monitor["top"], 1454 - monitor["left"]]

        #gadget color (1582, 802) rgb: 0 255 0
        # b2, g2, r2 = frame[802 - monitor["top"], 1582 - monitor["left"]]


        #for fps
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        

        #finding player/enemy
        for box in results.boxes:
            cls_id = int(box.cls[0])
            class_name = results.names[cls_id]
            x, y, w, h = box.xywh[0]

            center_x = int(x.item())
            center_y = int(y.item())

            if class_name == "player":
                player_center = (center_x, center_y)
            elif class_name == "enemy":
                enemy_centers.append((center_x, center_y))

        if player_center:
            px, py = player_center

            #super area
            cv2.ellipse(annotated_frame, (px, py), (h_ellipse, v_ellipse), 0, 0, 360, (255, 0, 255), 2)

            #gadget area
            cv2.ellipse(annotated_frame, (px, py), (200, 160), 0, 0, 360, (255, 0, 0), 2)

            #tracking
            closest_enemy = None
            min_dist = float("inf")
            for ex, ey in enemy_centers:
                dist = ((ex - px)**2 + (ey - py)**2)**0.5
                if dist < min_dist:
                    min_dist = dist
                    closest_enemy = (ex, ey)


            # if is_in_ellipse(ex, ey, px, py, h_ellipse, v_ellipse):
            #         pyautogui.moveTo(monitor["left"] + ex, monitor["top"] + ey)
            #         pyautogui.rightClick()
            #         print("Firing!")

            #moving to enemy
            if closest_enemy:
                start = False
                ex, ey = closest_enemy
                dx, dy = ex - px, ey - py

                #fps slows down when moving so release keys when enemy in range
                if is_in_ellipse(ex, ey, px, py, 100, 100):
                    release_key("w")
                    release_key("a")
                    release_key("s")
                    release_key("d")
                    # if (b1, g1, r1) == (30, 193, 255):
                    #     pyautogui.press("shift")

                else:

                    #stop moving when enemy in range (moving slows fps of frame)
                    threshold = 20 
                    if dx > threshold:
                        hold_key("d")
                        release_key("a")
                    elif dx < -threshold:
                        hold_key("a")
                        release_key("d")
                    else:
                        release_key("a")
                        release_key("d")

                    if dy > threshold:
                        hold_key("s")
                        release_key("w")
                    elif dy < -threshold:
                        hold_key("w")
                        release_key("s")
                    else:
                        release_key("w")
                        release_key("s")

                #using basic/super/gadget attacks when in certain ranges 
                if is_in_ellipse(ex, ey, px, py, h_ellipse, v_ellipse):
                    pyautogui.press("space")
                    print("basic")
                    if np.array_equal(frame[479, 895], np.array([30, 193, 255])):
                        pyautogui.press("shift")
                        print("super")
                    if is_in_ellipse(ex, ey, px, py, 200, 160):
                        if np.array_equal(frame[536, 1023], np.array([14, 255, 14])):
                            pyautogui.press("f")
                            print("gadget")
            else:

                #move forward at beginning of game
                if(start):
                    hold_key("w")
        else:
            release_all_keys()


        # (chatgpt) showing fps of frame
        cv2.putText(annotated_frame, f"{fps:.1f} FPS", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)


        cv2.imshow("Bot View", annotated_frame)
        if cv2.waitKey(1) == 27:  #click escape to exit
            break

cv2.destroyAllWindows()

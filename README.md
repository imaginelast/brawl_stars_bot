# Brawl Stars Computer Vision Agent

A real-time computer vision agent that observes a Brawl Stars game running in BlueStacks, detects the player and enemies using a TensorRT-optimized YOLO model, reasons about their spatial relationships, and automatically controls movement and abilities.

The system forms a closed-loop perception and control pipeline:

```text
BlueStacks Game
      ↓
Screen Capture (mss)
      ↓
OpenCV Frame Processing
      ↓
YOLO / TensorRT Object Detection
      ↓
Player + Enemy Positions
      ↓
Spatial Reasoning
      ↓
Target Selection + Range Checks
      ↓
Movement / Attack Decisions
      ↓
PyAutoGUI Input
      ↺
Updated Game State
```

Rather than following a predetermined sequence of actions, the controller continuously observes the current game state and uses the latest detections to determine its next action.

## Overview

The main controller, `bot.py`, captures the BlueStacks game window in real time and runs a trained YOLO model exported to TensorRT.

The detector identifies two classes:

* `player` — the controlled Brawler
* `enemy` — an opposing Brawler

The controller converts these detections into screen-space coordinates and uses them to:

1. Identify the controlled player
2. Locate visible enemies
3. Select the closest enemy
4. Calculate the relative position of the target
5. Determine whether the target is within different action ranges
6. Move toward the target when necessary
7. Execute basic attacks
8. Check and activate Super/Gadget abilities
9. Repeat using the newly observed game state

---

# System Architecture

The agent can be divided into four major stages:

### Perception

```text
BlueStacks
    ↓
mss screen capture
    ↓
OpenCV frame conversion
    ↓
YOLO / TensorRT
    ↓
Player + Enemy detections
```

### State Representation

The bounding boxes produced by YOLO are reduced to center coordinates:

```text
Player  → (px, py)

Enemy  → (ex, ey)
```

These coordinates form the primary representation used by the controller.

### Decision Making

The controller uses:

* Euclidean distance
* relative x/y displacement
* configurable thresholds
* elliptical range checks
* enemy availability
* Super/Gadget UI state

to determine the next action.

### Control

Decisions are converted into:

* `W/A/S/D` movement
* Space for basic attack
* Shift for Super
* F for Gadget

using PyAutoGUI.

---

# 1. Screen Capture

The bot automatically locates the BlueStacks window using PyGetWindow:

```python
win = gw.getWindowsWithTitle("Bluestacks")[0]
```

It then constructs an `mss` monitor region from the window's position and dimensions:

```python
monitor = {
    "top": win.top,
    "left": win.left,
    "width": win.width,
    "height": win.height
}
```

Every iteration captures the current contents of that region:

```python
screen = np.array(sct.grab(monitor))
frame = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)
```

This provides the agent with a continuously updated representation of the game.

Using `mss` avoids the overhead of manually taking screenshots through a heavier image-capture pipeline and is well suited to a real-time vision loop.

---

# 2. YOLO Object Detection

The trained detector is loaded as a TensorRT engine:

```python
model = YOLO("runs/detect/train/weights/best.engine")
```

For every captured frame, the model performs inference:

```python
results = model.predict(
    frame,
    conf=0.4,
    verbose=False
)[0]
```

A confidence threshold of `0.4` filters out lower-confidence detections.

Each detection contains:

* bounding box
* class ID
* class name
* confidence

The controller only needs the position of the detected entities, so bounding boxes are converted to center coordinates.

---

# 3. Player and Enemy Localization

The controller separates YOLO detections into:

```text
player_center
enemy_centers
```

The player is represented by a single coordinate.

Multiple enemy detections can be stored simultaneously.

For example:

```text
Player
  ↓
(px, py)

Enemies
  ↓
(ex1, ey1)
(ex2, ey2)
(ex3, ey3)
```

This allows the controller to reason about multiple visible opponents.

---

# 4. Target Selection

When multiple enemies are visible, the controller selects the closest one.

For every detected enemy, it calculates Euclidean distance:

```text
distance =
sqrt((enemy_x - player_x)^2 +
     (enemy_y - player_y)^2)
```

The enemy with the minimum distance becomes the active target.

This gives the controller a simple deterministic targeting policy:

```text
Visible enemies
      ↓
Calculate player-to-enemy distances
      ↓
Choose minimum distance
      ↓
Current target
```

The target is then used by both the movement and attack systems.

---

# 5. Relative Position Reasoning

Once an enemy has been selected, the controller calculates:

```python
dx = ex - px
dy = ey - py
```

These values describe where the enemy is relative to the player.

For example:

```text
dx > 0 → enemy is to the right
dx < 0 → enemy is to the left

dy > 0 → enemy is below
dy < 0 → enemy is above
```

This relative representation allows the bot to make movement decisions without needing an explicit world coordinate system.

---

# 6. Elliptical Range Modeling

One of the main spatial-reasoning components is the use of elliptical regions around the player.

The controller implements:

```python
def is_in_ellipse(ex, ey, px, py, a, b):
    return ((ex - px) ** 2) / a ** 2 + \
           ((ey - py) ** 2) / b ** 2 <= 1
```

Mathematically:

```text
             (ex - px)^2   (ey - py)^2
             ------------ + ------------ ≤ 1
                   a²            b²
```

where:

* `(px, py)` is the player position
* `(ex, ey)` is the enemy position
* `a` is the horizontal radius
* `b` is the vertical radius

This is useful because the controller does not have to treat every action as a perfect circle around the player.

Different ellipse dimensions can represent different gameplay ranges.

---

# 7. Movement Controller

If the target is outside the close-range region, the bot attempts to move toward it.

The controller compares the target's position with the player's position.

### Horizontal movement

```text
Enemy right of player
        ↓
Hold D

Enemy left of player
        ↓
Hold A
```

### Vertical movement

```text
Enemy below player
        ↓
Hold S

Enemy above player
        ↓
Hold W
```

A positional threshold prevents excessive direction changes caused by very small differences in detected coordinates.

---

# 8. Stateful Keyboard Input

The controller maintains the state of all movement keys:

```python
held_keys = {
    "w": False,
    "a": False,
    "s": False,
    "d": False
}
```

Instead of repeatedly sending key-down events, the controller uses helper functions:

```text
hold_key()
release_key()
release_all_keys()
```

The logic is effectively:

```text
If key is already held
    → do nothing

If key is not held
    → send keyDown()
    → mark it as held
```

Likewise, a key is only released when the controller believes it is currently being held.

This provides persistent movement while avoiding unnecessary repeated input events.

---

# 9. Close-Range Movement Control

The bot uses a smaller ellipse as a stopping region.

When the enemy enters this region:

```text
Enemy enters close range
        ↓
Release W/A/S/D
        ↓
Stop movement
```

This is intentional because the controller wants to avoid continuously moving once the target is already close enough to attack.

The source code also notes that movement can reduce the observed processing rate, so stopping movement near the target helps keep the vision loop responsive.

---

# 10. Basic Attack

When the enemy is inside the configured attack range, the controller performs the basic attack:

```python
pyautogui.press("space")
```

This decision is driven by the detected enemy's spatial relationship to the player rather than by a fixed timing sequence.

---

# 11. Super Detection

The controller also checks whether the Super ability appears to be available.

Instead of running YOLO on the UI element, it samples a specific pixel location and compares the detected color against the expected UI color.

If the Super indicator is detected as available:

```python
pyautogui.press("shift")
```

This is an example of combining two different forms of computer vision:

```text
YOLO
→ dynamic game entities

Pixel inspection
→ fixed UI state
```

This avoids using an object detector for a small, predictable UI element.

---

# 12. Gadget Detection

The Gadget controller follows a similar approach.

It first checks whether the enemy is inside the Gadget's configured spatial range.

It then checks a fixed UI pixel for the Gadget availability state.

Only when both conditions are satisfied does the bot execute:

```python
pyautogui.press("f")
```

Conceptually:

```text
Enemy in Gadget range?
        │
        ├── No → Do nothing
        │
        └── Yes
             ↓
      Gadget available?
             │
             ├── No → Do nothing
             │
             └── Yes
                  ↓
              Press F
```

---

# 13. Closed-Loop Autonomous Control

The most important aspect of the project is that perception and control are connected in a continuous feedback loop.

```text
┌──────────────────┐
│ Capture Game     │
│ Frame            │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ YOLO Detection   │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Locate Player    │
│ + Enemies        │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Select Target    │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Spatial Reasoning│
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Movement /       │
│ Attack / Ability │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Game State       │
│ Changes          │
└────────┬─────────┘
         │
         └──────────────→ Capture Next Frame
```

This architecture means that the bot can react to changes in the game instead of simply replaying predetermined commands.

---

# 14. Performance Monitoring

The controller calculates FPS directly from the time between iterations:

```python
current_time = time.time()
fps = 1 / (current_time - prev_time)
prev_time = current_time
```

The resulting FPS is rendered onto the debug frame.

This provides a simple way to observe how quickly the perception/control loop is operating.

TensorRT is used for the deployed YOLO model to reduce inference overhead, while `mss` provides lightweight screen capture.

---

# 15. Debug Visualization

The controller generates an annotated view of the current game frame.

The debug output displays:

* YOLO detections
* player/enemy labels
* FPS
* player-centered attack/range ellipses

This makes it possible to visually inspect the relationship between:

```text
Detection
    ↓
Spatial model
    ↓
Controller decision
```

and makes debugging model or coordinate errors significantly easier.

---

# Development / Detection Tool

The repository also contains `live_detect2.py`.

This script isolates the perception portion of the project so that the TensorRT detector can be tested independently from the autonomous controller.

It provides:

* live screen capture
* TensorRT YOLO inference
* bounding-box visualization
* FPS measurement
* mouse-coordinate tracking
* pixel-color inspection

The mouse pixel inspector is particularly useful for identifying UI colors used by the Super and Gadget checks.

---

# Engineering Decisions

## TensorRT Deployment

The YOLO model is deployed through TensorRT rather than using a standard training checkpoint in the runtime loop.

For a real-time agent, reducing inference overhead is important because every additional delay increases the gap between the observed game state and the resulting action.

## Center-Based Representation

The controller converts detected bounding boxes into center coordinates.

This keeps the downstream logic simple because target selection and spatial reasoning primarily require position.

## Elliptical Range Modeling

Different actions can have different horizontal and vertical tolerances.

Using an ellipse provides more control than a single scalar distance threshold.

## Hybrid Computer Vision

The project uses:

* learned object detection for dynamic entities
* deterministic pixel classification for fixed UI indicators

This is a practical hybrid approach because not every visual problem requires a neural network.

## Stateful Input

Movement keys are tracked internally so that the controller can maintain continuous movement without repeatedly issuing redundant key-down events.

## Feedback Over Prediction

The bot acts on the current frame and then observes the game again rather than attempting to perfectly predict the game's future state.

This keeps the external game screen as the source of truth.

---

# Limitations

The current implementation is designed around a controlled BlueStacks environment.

### Fixed UI Coordinates

The Super and Gadget checks rely on fixed pixel coordinates.

Changing:

* BlueStacks window size
* game resolution
* scaling
* screen layout

may require recalibration.

### Model-Specific Classes

The detector is trained around the project's `player` and `enemy` classes and is not a general-purpose Brawl Stars detector.

### Simple Target Selection

When multiple enemies are visible, the closest enemy is selected.

The controller does not currently consider:

* enemy health
* attack danger
* teammate position
* enemy abilities
* strategic positioning
* cover
* map objectives

### No Path Planning

Movement is based on relative x/y displacement.

The controller does not currently build a map or reason about obstacles.

### No Enemy Prediction

The controller reacts to the enemy's current detected position rather than explicitly predicting where the enemy will move.

### Fixed UI Detection

Super and Gadget availability are detected using specific pixel locations and colors rather than a generalized UI detector.

### Detection Errors

Incorrect or missing YOLO detections can cause incorrect movement, targeting, or attacks.

---

# Potential Improvements

Future versions could improve the system by adding:

* temporal smoothing of detections
* object tracking between frames
* enemy trajectory prediction
* lead-based aiming
* confidence-aware target selection
* health-aware target prioritization
* obstacle-aware path planning
* map-aware navigation
* action cooldown tracking
* automatic UI calibration
* learned movement policies
* multi-enemy tactical prioritization
* automatic UI-element detection
* more sophisticated decision-making based on game state

---

# Technologies

* Python
* YOLO
* Ultralytics
* TensorRT
* OpenCV
* NumPy
* mss
* PyAutoGUI
* PyGetWindow

---

# Project Takeaway

This project combines real-time computer vision, spatial reasoning, and automated control into a single feedback-driven system.

The key engineering challenge is not simply detecting an enemy. The detection has to be converted into a spatial representation, interpreted relative to the player, used to determine an appropriate action, and then translated into low-level keyboard input.

The resulting system connects:

**visual perception → spatial reasoning → decision making → automated control**

while using TensorRT, lightweight screen capture, and stateful input handling to keep the control loop responsive.

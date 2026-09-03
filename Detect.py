from ultralytics import YOLO
import cv2
import numpy as np
from vpython import canvas, box, vector, color, curve
import math



cam = {}
cameraArmed = False # Mostly unused, but can be used to toggle image camera without restarting the script
cam = cv2.VideoCapture(1) # Camera to use
model = YOLO("yolov5s.pt") # Model to use
model.overrides["imgsz"] = 1280 # Resolution to pass to YOLO, higher number = more compute cost but further vision range/accuracy

horizontal_fov = 81 # Horizontal FOV of the camera, used to output theta angles
vertical_fov = 51 # Vertical FOV of camera, used to output rho angles
horizontal_pixels = 1920 # Horizontal pixel count of camera
vertical_pixels = 1080 # Vertical pixel count of camera
 

# For the 3d renderer
show_render = False
DEFAULT_DISTANCE = 10  # arbitrary scene units
_scene = None
_active_boxes = []



# Entirely optional, could allow for a startup animation? Maybe PROVE logo?
def startupAnimation():
    startupAnimationPath = "startup.mp4"
    cap = cv2.VideoCapture(startupAnimationPath)
    if not cap.isOpened():
        print("Error: Could not open startup animation")
    while cap.isOpened():
        ret, startupframe = cap.read()
        if not ret:
            break
        cv2.imshow(startupframe)

# Inputs camera data, applies a yellow filter
def yellowMask(inputFrame):
    hsv = cv2.cvtColor(inputFrame, cv2.COLOR_BGR2HSV)
        
        # HSV bounds for yellow (optimized for typical indoor webcam lighting)
    lower_yellow = np.array([18, 94, 140])
    upper_yellow = np.array([30, 255, 255])
    
    # Create the binary yellow mask
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Filter the frame to show ONLY yellow pixels
    yellow_isolated = cv2.bitwise_and(inputFrame, inputFrame, mask=yellow_mask)
    return yellow_isolated


# Takes a yellow isolated frame, outputs road lines
# WIP
def refineRoadLines(yellow_isolated, drawFrame, row_step=3, sigma_frac = 0.25, floor = 0.15):
    # collapse to a single-channel mask so we can sum columns per band
    gray_mask = cv2.cvtColor(yellow_isolated, cv2.COLOR_BGR2GRAY)
    h, w = gray_mask.shape[:2]

    points = []

    xs = np.arange(w, dtype=np.float64)
    sigma = w * sigma_frac
    wx = np.exp(-0.5 * ((xs - w / 2) / sigma) ** 2)
    wx = floor + (1.0 - floor) * wx

    for y in range(0, h, row_step):
        band = gray_mask[y:y + row_step, :]

        # how much yellow "mass" is in each column of this band
        col_sums = np.sum(band, axis=0).astype(np.float64) * wx 
        total = col_sums.sum()

        if total <= 10:
            continue  # no yellow in this band, skip it

        # weighted centroid = center of mass along x
        xs = np.arange(w, dtype=np.float64)
        cx = int(np.sum(xs * col_sums) / total)
        cy = y + row_step // 2

        points.append((cx, cy))

    # connect the points into a line
    if len(points) >= 2:
        pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(drawFrame, [pts], isClosed=False, color=(0, 0, 255), thickness=2)

    for p in points:
        cv2.circle(drawFrame, p, 3, (0, 255, 0), -1)

    return drawFrame, points

# Creates the VPython Scene, for visualizing nearby area
def _get_scene():
    global _scene
    if _scene is None:
        _scene = canvas(title='Surrounding Area', width=800, height=600,
                         background=color.black, forward=vector(0, 0, -1))
        box(canvas=_scene, pos=vector(0, 0, 0), size=vector(0.3, 0.3, 0.6), color=color.white)
    return _scene

# Draws detected objects onto VPython scene, using default distance from camera
def drawSurroundingArea(boundingBoxes, distance=DEFAULT_DISTANCE):
    """
    boundingBoxes: list of (theta1, theta2, rho1, rho2) tuples in degrees,
                   theta = horizontal angle, rho = vertical angle, 0 = straight ahead.
    distance: how far out to place the projection plane, since angle-only
              detections carry no depth information.
    """
    scene = _get_scene()

    # clear last frame's boxes/lines before drawing this frame's
    for b in _active_boxes:
        b.visible = False
    _active_boxes.clear()

    for theta1, theta2, rho1, rho2 in boundingBoxes:
        t1, t2 = math.radians(theta1), math.radians(theta2)
        r1, r2 = math.radians(rho1), math.radians(rho2)

        x1, x2 = distance * math.tan(t1), distance * math.tan(t2)
        y1, y2 = distance * math.tan(r1), distance * math.tan(r2)

        center = vector((x1 + x2) / 2, (y1 + y2) / 2, -distance)
        size = vector(abs(x2 - x1), abs(y2 - y1), 0.05)

        b = box(canvas=scene, pos=center, size=size, color=color.red, opacity=0.6)
        _active_boxes.append(b)

        # detection line from the camera origin to this box's center of mass
        line = curve(canvas=scene, pos=[vector(0, 0, 0), center], color=color.yellow, radius=0.02)
        _active_boxes.append(line)

# Closes the 3d scene on shutdown
def closeSurroundingArea():
    global _scene
    if _scene is not None:
        _scene.delete()
        _scene = None

# Most of the logic is contained here
def detectLoop():
    if not cam.isOpened():
        print("Error: Could not open camera")
        return
    cameraArmed = True
    framenum = 0
    
    while cameraArmed and cam.isOpened():
        ret, frame = cam.read()

        if not ret:
            cameraArmed = False
            break
        results = model(frame)

        outlines = results[0].boxes.xyxy
        
        annotated_frame = results[0].plot()
        yellow_isolated = yellowMask(frame)
        road_lines_frame, points = refineRoadLines(yellow_isolated, frame.copy())
        processed_data = []
        for bounding_box in outlines:
            x1, y1, x2, y2 = bounding_box
            theta1 = (x1/horizontal_pixels-0.5) * horizontal_fov
            theta2 = (x2/horizontal_pixels-0.5) * horizontal_fov
            rho1 = (0.5-y1/vertical_pixels) * vertical_fov
            rho2 = (0.5-y2/vertical_pixels) * vertical_fov
            processed_data.append((theta1, theta2, rho1, rho2))
        if(show_render):
            drawSurroundingArea(processed_data)

        cv2.imshow('road line', road_lines_frame)
        cv2.imshow('yellow', yellow_isolated)
        cv2.imshow('raw', frame)
        cv2.imshow('annotated', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cam.release()
    cv2.destroyAllWindows()
    if(show_render):
        closeSurroundingArea()

# startupAnimation()
detectLoop()
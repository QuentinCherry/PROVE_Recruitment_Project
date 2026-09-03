import cv2
import numpy as np

# 1. Open the live webcam (0 is typically the default built-in camera)
cap = cv2.VideoCapture(1)

# Check if webcam opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam started. Hold a yellow object up to the camera!")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # --- Step A: Isolate Yellow Color ---
    # Convert webcam frame to HSV space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # HSV bounds for yellow (optimized for typical indoor webcam lighting)
    lower_yellow = np.array([18, 94, 140])
    upper_yellow = np.array([30, 255, 255])
    
    # Create the binary yellow mask
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Filter the frame to show ONLY yellow pixels
    yellow_isolated = cv2.bitwise_and(frame, frame, mask=yellow_mask)

    # --- Step B: Edge Detection on Yellow Elements ---
    # Convert to grayscale
    gray_yellow = cv2.cvtColor(yellow_isolated, cv2.COLOR_BGR2GRAY)
    
    # Apply a light blur to smooth out webcam noise or camera grain
    blurred = cv2.GaussianBlur(gray_yellow, (5, 5), 0)
    
    # Trace the edges of the yellow shapes
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

    # --- Step C: Create Green Overlay on Live Feed ---
    overlay = frame.copy()
    overlay[edges == 255] = [0, 255, 0]  # Color the edge pixels bright green (BGR format)

    # --- Step D: Display Streams ---
    cv2.imshow('Line Edges', edges)
    cv2.imshow('Yellow Isolation', yellow_isolated)
    cv2.imshow('Edges Overlaid (Green)', overlay)

    # Press 'q' to exit the stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()

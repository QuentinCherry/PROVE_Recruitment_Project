import cv2

cam = cv2.VideoCapture(1)
if not cam.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # edges = cv2.Canny(gray, 100, 200)

    cv2.imshow('video', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



cam.release()
cv2.destroyAllWindows()
import cv2

cap = cv2.VideoCapture(0)
scaling_factor = 0.5


while True:
    
    ret, frame = cap.read()

    frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

    cv2.imshow('Webcam', frame)

    c = cv2.waitKey(1)

    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()
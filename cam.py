import cv2

cam = cv2.VideoCapture(0)
while True:
    ret , frame = cam.read()
    cv2.imshow("cam",frame)
    x = cv2.waitKey(1) & 0xff
    if x == 27:
        break

cam.release()
cv2.destroyAllWindows()
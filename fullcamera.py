#to activate Virtualenv ir na pasta Venv e ativar source bin/activate

import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt
from skimage.measure import  compare_ssim as ssim

  #http://<user>:<password>@<ip>:<port>/cgi-bin/snapshot.cgi?)

def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err


def selectRoi(frame):
    r = cv2.selectROI(frame)
    imCrop = frame[int(r[1]):int(r[1]+r[3]),int(r[0]):int(r[0]+r[2])]
    cv2.imshow("cropped",imCrop)
    return r

def rectangle(frame):
    cv2.rectangle(overlay,(450,100),(1150,500),(0,0,0),-1)
    cv2.addWeighed(overlay,alpha,output,1-alpha,0,output)


# capture = cv2.VideoCapture("http://admin:admin@192.168.2.2:8080/cgi-bin/snapshot.cgi?")
capture = cv2.VideoCapture("rtsp://user:1@192.168.2.6:554/cam/realmonitor?channel=1&subtype=0")
backgrd = cv2.createBackgroundSubtractorMOG2()


kernel = np.ones((5,5),np.float32)/25

template1 = cv2.imread("template1Full.png")
template1 = cv2.cvtColor(template1,cv2.COLOR_BGR2GRAY)
template1blur = cv2.GaussianBlur(template1, (21, 21), 0)
clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
template1clahe = clahe.apply(template1)
template1clahe = cv2.GaussianBlur(template1clahe, (21, 21), 0)
templateframe = template1clahe
initial = 0;
naVaga = 0;
saiuVaga = 0;
cont2 = 0;
cont = 0;
flag = False;
secondcont = 0;
# show = False;
countpick = 400;
while True:
    text = "vaga vazia"
    ret, frame = capture.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    s1 = ssim(gray,template1clahe)

    if(s1<0.8):
        frameDelta = cv2.absdiff(gray,templateframe)
        thresh = cv2.threshold(frameDelta,25,255,cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        image,contours,hierarchy= cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) > 500:
                continue
            (x,y,w,h) =  cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = "vaga cheia"
        # #  
    if(s1>0.8):
        templateframe = gray
    cv2.imshow("frame",frame)


    x = cv2.waitKey(10) & 0xff    
    if(x==27):
        break
    
    if(x==32):
        cont = cont+1
        cv2.imwrite(str(cont)+"full.png",frame)
        cv2.imwrite(str(cont)+"roi.png",roi)

capture.release()
cv2.destroyAllWindows()
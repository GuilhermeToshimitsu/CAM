# import base64
# import time
# import urllib
#to activate Virtualenv ir na pasta Venv e ativar source bin/activate

import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt
from skimage.measure import  compare_ssim as ssim
import time

# from cv2 import GetSubRect


"""
Examples of objects for image frame aquisition from both IP and
physically connected cameras
Requires:
 - opencv (cv2 bindings)
 - numpy
"""


  #http://<user>:<password>@<ip>:<port>/cgi-bin/snapshot.cgi?)
# camera= ipCamera(

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
# 
capture = cv2.VideoCapture("rtsp://user:1@192.168.2.6:554/cam/realmonitor?channel=1&subtype=0")
backgrd = cv2.createBackgroundSubtractorMOG2()


kernel = np.ones((5,5),np.float32)/25

template1 = cv2.imread("template3.png")
template1 = cv2.cvtColor(template1,cv2.COLOR_BGR2GRAY)
template1blur = cv2.GaussianBlur(template1, (21, 21), 0)
template2 = cv2.imread("template2.png")
template2 = cv2.cvtColor(template2,cv2.COLOR_BGR2GRAY)
template2blur = cv2.GaussianBlur(template2, (21, 21), 0)
clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
template1clahe = clahe.apply(template1)
template1clahe = cv2.GaussianBlur(template1clahe, (21, 21), 0)
template2clahe = clahe.apply(template2)
template2clahe = cv2.GaussianBlur(template2clahe, (21, 21), 0)
templateframe = template2clahe
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
    start = time.time()
    ret, frame = capture.read()
    if not ret:
        break
    
    #region of interest(of car)
    roi = frame[114:410,730:1150]
    roi2 = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    roi2 = clahe.apply(roi2)
    roi2blur = cv2.GaussianBlur(roi2, (21, 21), 0)

    s1 = ssim(roi2blur,template1clahe)
    s2 = ssim(roi2blur,template2clahe)

    print(s1)
    print(s2)

    frameDelta = cv2.absdiff(roi2blur,templateframe)
    thresh = cv2.threshold(frameDelta,25,255,cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    # contours1= cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow("thresh",thresh)
    # cv2.imshow("clahetemplate",template2clahe)
    cv2.imshow("roi",roi)
    # cv2.imshow("roi2blur",roi2blur)
    cv2.imshow("delta",frameDelta)
    n = np.sum(thresh==255)  
    if(n>50000):
        print("Carro na vaga")
    else:
        print("Sem carro na vaga")

    if(s1>0.8 or s2>0.8):
        print("changeframe")
        templateframe =roi2blur
        cv2.imshow("template", templateframe)
    end = time.time()
    print(end-start)

    #apply car

    # print(text)
    
    # mask = backgrd.apply(roi)
    # n = np.sum(mask==255)
    # frame[100:500,450:1150,2]=roi2
    # cv2.imshow("webcam",frame)  

    # cv2.imshow("backgroud",mask)
    # cv2.imshow("imCrop",imCrop)
    # cv2.imshow("roi",roi)
    # cv2.imshow("template1clahe",template1clahe)
    # cv2.imshow("roi2",roi2)
    # cv2.imshow("roi2blur",roi2blur)
    # cv2.imshow("template1",template1)
    # cv2.imshow("template1blur",template1blur)
    # cv2.imshow("thresh",thresh2)
    # cv2.imshow("thresh1",thresh1)
    # cv2.imshow("delta",frameDelta2)
    # cv2.imshow("delta1",frameDelta1)
    # print("n: "+str(n))
    # print("threshhold: "+ str(countpick))
  
    # s2 = ssim(roi2,template2)
    # s3 = ssim(roi2,template3)

    # if(s1>0.75 or s2>0.75 or s3>0.75):
    #     flag=False;

    # if(n>10000):
    #     if(s1<0.5 or s2<0.5 or s3<0.5):
    #         flag = True;
    #         if(secondcont<10):
    #             cv2.imwrite(str(secondcont)+"carparked.png",roi)
    #             secondcont=secondcont+1
    #     if(s1>0.8 and s2>0.8 and s3>0.8):
    #         flag = False;

    # print(flag)
        
    #     # cont2 =cont2+1
    #     # cv2.imwrite(str(countpick)+"carro.png",frame)
    #     # cv2.imwrite(str(countpick)+"carroroi.png",roi)
    #     # cv2.imwrite(str(countpick)+"carroroibg.png",mask)
    #     # countpick = countpick + 400

    x = cv2.waitKey(10) & 0xff    
    if(x==27):
        break
    
    if(x==32):
        cont = cont+1
        cv2.imwrite(str(cont  )+".png",frame)
        cv2.imwrite(str(cont)+"roi.png",roi)
# # else: 
# #     print("not open")

capture.release()
cv2.destroyAllWindows()
import imutils
import datetime
from threading import Thread, Lock
import cv2
import argparse
from skimage.measure import compare_ssim as ssim
import time
import numpy as np
from darkflow.net.build import TFNet

# @article{yolov3,
# title={YOLOv3: An Incremental Improvement},
# author={Redmon, Joseph and Farhadi, Ali},
# journal = {arXiv},
# year={2018}
# }

class CAM:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grab, self.frame) = self.stream.read()
        self.stopped = False
        self.thread = 0

    def start(self):
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            (self.grab, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

    def release(self):
        self.stream.release()


class Finder:
    def __init__(self, vs, usesubframe, disablesubframe, x1, y1, x2, y2, usetemplate,showsubframe,showdelta,showtemplate,log,savephotos,labels,shows1,threshold,showtime):
        self.vs = vs.start()
        self.shows1 = shows1
        self.showtime = showtime
        self.bgr = cv2.createBackgroundSubtractorMOG2()
        self.bgr2 = 0
        self.savephotos = savephotos
        self.labels = labels
        self.showdelta = showdelta
        self.showtemplate = showtemplate
        self.usesubframe = usesubframe
        self.usetemplate = usetemplate
        self.showsubframe =  showsubframe
        self.contador = 0
        self.log = log
        if self.log:
            self.file = open("log.txt","w+")
        if(usesubframe):
            self.x1 = x1
            self.x2 = x2
            self.y1 = y1
            self.y2 = y2
        else:
            self.x1 = 114
            self.x2 = 410
            self.y1 = 730
            self.y2 = 1150
        self.disablesubframe = disablesubframe
        self.template1 = cv2.imread("templates/template1.png")
        self.template2 = cv2.imread("templates/template2.png")
        self.template3 = cv2.imread("templates/template3.png")
        self.timestart = datetime.datetime.now()
        self.thresh = 0
        self.frameDelta = 0
        self.s1 = 0
        self.option = {'model': 'cfg/yolov2-tiny.cfg',
                       'load': 'bin/yolov2-tiny.weights', 'threshold': float(threshold), 'gpu': 1.0}
        # self.option = {'model': 'cfg/yolov2-tiny-voc-2c.cfg',
        #                'load': 9500, 'threshold': 0.01, 'gpu': 1.0}
        self.tfnet = TFNet(self.option)
        self.frame = vs.read()
        self.specialframe = 0
        self.colors = [tuple(255*np.random.rand(3)) for i in range(5)]
        if self.usetemplate:
            if(self.timestart.hour > 17):
                self.template = cv2.cvtColor(self.template2, cv2.COLOR_BGR2GRAY)
                print("template2")
            elif(self.timestart.hour > 9):
                self.template = cv2.cvtColor(self.template3, cv2.COLOR_BGR2GRAY)
                print("template3")
            else:
                self.template = cv2.cvtColor(self.template1, cv2.COLOR_BGR2GRAY)
                print("template1")
        else:
            if not disablesubframe:
                self.template = cv2.cvtColor(self.frame[self.x1:self.x2, self.y1:self.y2], cv2.COLOR_BGR2GRAY)
            else:
                self.template = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.templateclahe = self.clahe.apply(self.template)
        self.templateblur = cv2.GaussianBlur(self.templateclahe, (21, 21), 0)
        self.templateframe = self.templateblur
        self.img = self.templateblur
        self.stopped = False
        self.yoloLock = Lock()
        self.cadeado = 0
        self.results = []
        self.valid = []
        self.primeiro = 0
        self.doYolo = 0

    def start(self):
        y = Thread(target=self.check, args=())
        x = Thread(target=self.update, args=())
        z = Thread(target=self.yolo, args=())
        x.start()
        y.start()
        z.start()

    def stop(self):
        self.stopped = True
        if self.log:
            self.file.close()
        self.vs.stop()
    #update -> cam read
    def update(self):
        try:
            while True:
                if self.stopped:
                    return
                self.frame = self.vs.read()
                if(self.primeiro == 0):
                    self.specialframe = self.frame[:]
                else:
                    if not self.valid:
                        self.specialframe = self.frame[:]
                    else:
                        for color, result in zip(self.colors, self.results):
                            t1 = (result['topleft']['x'], result['topleft']['y'])
                            br = (result['bottomright']['x'],
                                result['bottomright']['y'])
                            label = result['label']
                            confidence = result['confidence']        
                            if(label == "car" or label =="truck" or label == 'train' or label=='bus'):
                                label = "carro"
                            if  label == "suitcase":
                                label = 'carro' 
                            if label == "person":
                                if confidence < 0.7:
                                    t1=(0,0)
                                    br =(0,0)    
                            if(label == "bottle" or label == "TV "):
                                print("notlabel")            
                            self.specialframe = cv2.rectangle(self.frame, t1, br, color, 7)
                            self.specialframe = cv2.putText(self.frame, label, t1, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
                cv2.imshow("img1", self.specialframe)
                if self.showsubframe:
                    cv2.imshow("subframe",self.frame[self.x1:self.x2, self.y1:self.y2])
                if self.showdelta:
                    cv2.imshow("Delta",self.frameDelta)
                    cv2.imshow("Threshold Motion",self.thresh)
                    cv2.imshow("bgr2",self.bgr2)
                if self.showtemplate:
                    cv2.imshow("template",self.templateframe)
                x = cv2.waitKey(10) & 0xff
                if(x == 27):
                    self.stop()
                    cv2.destroyAllWindows()
                    break
                if(x == 32):
                    self.cadeado = 1
                    print("manual change template")
                    if self.disablesubframe:
                        self.img = self.frame[:]
                    else:
                        self.img = self.frame[self.x1:self.x2, self.y1:self.y2]
                    self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
                    self.img = self.clahe.apply(self.img)
                    self.img = cv2.GaussianBlur(self.img, (21, 21), 0)
                    self.templateframe = self.img
                    self.cadeado = 0
                    data = datetime.datetime.now()
                    cv2.imwrite(str(data)+".png",self.frame)
        except Exception as e:
            print("error thread show (x) :" + str(e))

    def check(self):
        try:
            while True:
                if self.stopped:
                    return
                # start = datetime.datetime.now()
                if(self.cadeado == 0):
                    if self.disablesubframe:
                        img1 = self.frame[:]
                    else:
                        img1 = self.frame[self.x1:self.x2, self.y1:self.y2]
                    startbgr1 = datetime.datetime.now()
                    self.img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                    self.img = self.clahe.apply(self.img)
                    self.img = cv2.GaussianBlur(self.img, (21, 21), 0)
                    self.s1 = ssim(self.img, self.templateframe)
                    self.frameDelta = cv2.absdiff(self.img,self.templateframe)
                    self.thresh = cv2.threshold(self.frameDelta,25,255,cv2.THRESH_BINARY)[1]
                    self.thresh = cv2.dilate(self.thresh, None, iterations=2)
                    endbgr1 = datetime.datetime.now()
                    startbgr2 = datetime.datetime.now()
                    self.bgr2 = self.bgr.apply(img1)
                    endbgr2 = datetime.datetime.now()
                    if(self.showtime):
                        print("tempo de BGR1: " + str((endbgr1-startbgr1).total_seconds()))
                        print("tempo de BGR2: " + str((endbgr2-startbgr2).total_seconds()))
                    if(self.shows1):
                        print(self.s1)
                    if(self.s1 > 0.9):
                        self.templateframe = self.img
                        # print("troca template")
                    else:
                        if not self.results:
                            self.templateframe = self.img
                    # print(self.s1)
                    if(self.s1 < 0.8):
                        print("call yolo!")
                        if self.doYolo == 0:
                            self.doYolo = 1
                            self.contador = 10
        except Exception as e:
            print("error thread check (y) : "+str(e))
                

    def yolo(self):
        try:
            while True:
                if self.stopped:
                    return
                cont = 0
                while self.doYolo == 0:
                    cont += 1
                    if self.stopped:
                        return
                    #activate yolo every 1 seconds
                    time.sleep(0.1)
                    if cont > 10:
                        print("self activation yolo")
                        self.doYolo = 1
                nowframe = self.frame[:]
                startyolo = datetime.datetime.now()
                self.results = self.tfnet.return_predict(nowframe)
                endyolo = datetime.datetime.now()
                if self.showtime:
                    print("Yolo Time: "+str((endyolo-startyolo).total_seconds()))
                self.doYolo = 0
                print(self.results)
                self.contador -=1
                self.yoloLock.acquire()
                self.valid = []
                for result in self.results:
                    if self.savephotos:
                        for color, result in zip(self.colors, self.results):
                            t1 = (result['topleft']['x'], result['topleft']['y'])
                            br = (result['bottomright']['x'],
                                result['bottomright']['y'])
                            label = result['label']
                            confidence = result['confidence']
                            if(label =='person'):
                                label = "pedestre"
                            if(label == 'car'):
                                label = 'carro'
                            if(label == 'suitcase'):
                                label == 'carro'
                            if(label == 'truck'):
                                label = 'carro'
                            if label == 'carro':
                                yoloframe = cv2.rectangle(nowframe, t1, br, color, 7)
                                yoloframe = cv2.putText(nowframe, label, t1, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
                            else:
                                print(0)
                        tempo = datetime.datetime.now()
                        cv2.imwrite("train/dados/"+str(tempo)+"1.jpg",yoloframe)
                    if(result['label'] == 'person' or result['label'] == 'car' or result['label'] == 'bus' or result['label'] == 'motorbike' or result['label'] == 'truck' or result['label'] == 'bike'):
                        self.valid = result
                self.yoloLock.release()
                hora = datetime.datetime.now()
                if self.log:
                    if self.valid:
                        self.file.write(str(hora)+" : "+str(self.valid)+'\n')
                self.primeiro = 1
        except Exception as e:
            print("error thread yolo (z): "+str(e))

    def read(self):
        return self.vs.frame

    def showTemplateClahe(self):
        return self.templateclahe

    def showTemplate(self):
        return self.template

    def showTemplateFrame(self):
        return self.templateframe

    def returnssim(self):
        return self.s1

def Main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-r","--source",help ="choose the source", default="rtsp://user:1@192.168.1.254:554/cam/realmonitor?channel=1&subtype=0",action="store",dest="src")
    ap.add_argument("-f","--roiframe",help="choose the ROI spot for detection",action="store_true")
    ap.add_argument("-x","--disablesubframe",help="Apply detection to whole screen isntead of subframe",action="store_true")
    ap.add_argument("-t","--usetemplate",help="use default templates",action="store_true")
    ap.add_argument("-s","--showsubframe",help = "show subframe",action="store_true")
    ap.add_argument("-i","--information",help="show status of opencv",action="store_true")
    ap.add_argument("-d","--showdelta",help="delta and move threshold",action="store_true")
    ap.add_argument("-y","--showtemplate",help="show template for bgr",action="store_true")
    ap.add_argument("-l","--storelog",help="show log for yolo call results by time",action="store_true")
    ap.add_argument("-p","--savephotos", help ="save photos detected to use for further training",action="store_true")
    ap.add_argument("-a","--savelabels", help="choose the label to save the new models, ex: -a car_bus_truck, will save only photos of detected car bus or train",action="store",dest="labels")
    ap.add_argument("-w","--prints1",help="print all logs of s1",action="store_true")
    ap.add_argument("-o","--threshold",help= "choose your threshold for Yolo", action="store",default=0.6,dest="threshold")
    ap.add_argument("-k","--showtime",help="show time of functions",action="store_true")
    args = ap.parse_args()
    default1="rtsp://user:1@192.168.1.254:554/cam/realmonitor?channel=1&subtype=0"
    if args.labels:
        args.savephotos = True
        args.labels = args.labels.split("_")

    if args.src.isdigit():
        args.src = int(args.src)
        args.roiframe = True
        args.usetemplate = False
    if(args.disablesubframe):
        args.showsubframe = False
        args.roiframe = False
    cam = cv2.VideoCapture(args.src)
    if cam.isOpened and args.roiframe:
        ret,frame = cam.read()
        args.usetemplate = False
        r = cv2.selectROI(frame)
        cv2.destroyAllWindows()
    cam.release()
    if args.roiframe:
        subframex1 = int(r[1])
        subframey1 = int(r[0])
        subframex2 = int(r[1]+r[3])
        subframey2 = int(r[0]+r[2])
    else:
        subframex1 = 0
        subframex2 = 0
        subframey1 = 0
        subframey2 = 0
    vs = CAM(args.src)

    if args.information:
	    print(cv2.getBuildInformation())
    xs = Finder(vs,args.roiframe,args.disablesubframe,subframex1,subframey1,subframex2,subframey2,args.usetemplate,args.showsubframe,args.showdelta,args.showtemplate,args.storelog,args.savephotos,args.labels,args.prints1,args.threshold,args.showtime).start()

if __name__ == '__main__':
	Main()

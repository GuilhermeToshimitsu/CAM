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


class FPS:
    def __init__(self):
        self._start = None
        self._end = None
        self._Frames = 0

    def start(self):
        self.start = datetime.datetime.now()
        return self

    def stop(self):
        self._end = datetime.datetime.now()

    def update(self):
        self._Frames += 1

    def elapsed(self):
        return (self._end - self._start).total_seconds()

    def fps(self):
        return self._Frames/self.elapsed()


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
    def __init__(self, vs, usesubframe, disablesubframe, x1, y1, x2, y2, usetemplate,showsubframe,showdelta,showtemplate,log,savephotos,labels):
        self.vs = vs.start()
        self.savephotos = savephotos
        self.labels = labels
        self.showdelta = showdelta
        self.showtemplate = showtemplate
        self.usesubframe = usesubframe
        self.usetemplate = usetemplate
        self.showsubframe =  showsubframe
        self.contador = 0
        self.log = log
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
                       'load': 'bin/yolov2-tiny.weights', 'threshold': 0.6, 'gpu': 0.0}
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
                    self.specialframe = self.frame
                else:
                    if not self.valid:
                        self.specialframe = self.frame
                    else:
                        # print("aqui")
                        self.cadeado2 = 1
                        for color, result in zip(self.colors, self.results):
                            t1 = (result['topleft']['x'], result['topleft']['y'])
                            br = (result['bottomright']['x'],
                                result['bottomright']['y'])
                            label = result['label']
                            confidence = result['confidence']                            
                            self.specialframe = cv2.rectangle(self.frame, t1, br, color, 7)
                            self.specialframe = cv2.putText(self.frame, label, t1, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
                        self.cadeado2 = 0
                cv2.imshow("img1", self.specialframe)
                if self.showsubframe:
                    cv2.imshow("subframe",self.frame[self.x1:self.x2, self.y1:self.y2])
                if self.showdelta:
                    cv2.imshow("Delta",self.frameDelta)
                    cv2.imshow("Threshold Motion",self.thresh)
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
                        self.img = self.frame
                    else:
                        self.img = self.frame[self.x1:self.x2, self.y1:self.y2]
                    self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
                    self.img = self.clahe.apply(self.img)
                    self.img = cv2.GaussianBlur(self.img, (21, 21), 0)
                    self.templateframe = self.img
                    self.cadeado = 0
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
                        self.img = self.frame
                    else:
                        self.img = self.frame[self.x1:self.x2, self.y1:self.y2]
                    self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
                    self.img = self.clahe.apply(self.img)
                    self.img = cv2.GaussianBlur(self.img, (21, 21), 0)
                    self.s1 = ssim(self.img, self.templateframe)
                    self.frameDelta = cv2.absdiff(self.img,self.templateframe)
                    self.thresh = cv2.threshold(self.frameDelta,25,255,cv2.THRESH_BINARY)[1]
                    self.thresh = cv2.dilate(self.thresh, None, iterations=2)
                    if(self.s1 > 0.9):
                        self.templateframe = self.img
                        # print("troca template")
                    else:
                        if not self.results:
                            self.templateframe = self.img
                    # print(self.s1)
                    if(self.s1 < 0.8):
                        # print("call yolo!")
                        if self.doYolo == 0:
                            self.doYolo = 1
                            # self.contador = 10
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
                    time.sleep(0.1)
                    if cont > 10:
                        print("self activation yolo")
                        self.doYolo = 1
                self.results = self.tfnet.return_predict(self.frame)
                self.doYolo = 0
                self.contador -=1
                self.yoloLock.acquire()
                self.valid = []
                for result in self.results:
                    if self.savephotos:
                        if(self.contador<1 or cont>10):
                            print("aqui")
                            #save photos at a counter (avoid saving to many photos)
                            if self.labels:
                                for i in self.labels:
                                    if result['label']==i:
                                        s1time = datetime.datetime.now()
                                        mini = self.frame[result['topleft']['x']:result['bottomright']['x'],result['topleft']['y']:result['bottomright']['y']]
                                        cv2.imwrite("train/"+result['label']+":"+str(s1time)+".jpg",mini)
                            else:
                                s2time = datetime.datetime.now()
                                mini = self.frame[result['topleft']['x']:result['bottomright']['x'],result['topleft']['y']:result['bottomright']['y']]
                                cv2.imwrite("train/"+result['label']+":"+str(s2time)+".jpg",mini)
                            self.contador = 10
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

    def readFrame(self):
        self.frame = self.vs.read()
        self.img = self.frame[114:410, 730:1150]
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.img = self.clahe.apply(self.img)
        self.img = cv2.GaussianBlur(self.img, (21, 21), 0)
        return self.img

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
    # print(subframex1)
    # print(subframex2)
    # print(subframey1)
    # print(subframey2)
    # else:
	# 	print("camera nao conectada")
	# 	return
	# vs = CAM(src ="rtsp://user:1@192.168.2.6:554/cam/realmonitor?channel=1&subtype=0" ).start()
	# print(args.src)
    # v1 = cv2.VideoCapture(1)
    print(args.src)
    print(type(args.src))
    # vs = cv2.VideoCapture(args.src)
    vs = CAM(args.src)
    # while True:
    #     frame = vs.read()
    #     cv2.imshow("v1",frame)
    #     x = cv2.waitKey(1) & 0xff
    #     if(x == 27):
    #         break   
    #         cv2.destroyAllWindows()
    # vs.release()

	# fps = FPS().start()
    if args.information:
	    print(cv2.getBuildInformation())
    xs = Finder(vs,args.roiframe,args.disablesubframe,subframex1,subframey1,subframex2,subframey2,args.usetemplate,args.showsubframe,args.showdelta,args.showtemplate,args.storelog,args.savephotos,args.labels).start()

if __name__ == '__main__':
	Main()

import datetime
import cv2
import numpy as np
from darkflow.net.build import TFNet
import psutil
import os
pid = os.getpid()
py = psutil.Process(pid)


option = {'model':'cfg/yolov2-tiny.cfg','load': 'bin/yolov2-tiny.weights','threshold': 0.6,'gpu':1.0}
tfnet = TFNet(option)
colors = [tuple(255*np.random.rand(3)) for i in range(5)]
capture = cv2.VideoCapture("rtsp://user:1@192.168.2.254:554/cam/realmonitor?channel=1&subtype=0")
capture.set(cv2.CAP_PROP_FRAME_WIDTH,640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
print("number of threads: ",psutil.cpu_count())
print("number of cores(physical) : ",psutil.cpu_count(logical=False))

while True:
    starttotal = datetime.datetime.now()
    ret, frame = capture.read()
    if not ret:
        break
    newframe = frame[100:500,650:1300]
        
    start = datetime.datetime.now()
    results = tfnet.return_predict(newframe)
    end = datetime.datetime.now()
    print("tempo de yolo: "+ str((end-start).total_seconds()))
    start = datetime.datetime.now()
    for color,result in zip(colors, results):
        t1 =(result['topleft']['x'],result['topleft']['y'])
        br =(result['bottomright']['x'],result['bottomright']['y'])
        label = result['label']
        confidence = result['confidence']
        specialframe = cv2.rectangle(newframe,t1,br,color,7)
        specialframe = cv2.putText(newframe,label+" :"+str(confidence),t1,cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)
    end = datetime.datetime.now()
    cv2.imshow("img1",frame)
    x = cv2.waitKey(10) & 0xff
    if(x==27):
        break
    endtotal = datetime.datetime.now()
    print("tempo de loop: "+ str((endtotal-starttotal).total_seconds()))
    memoryUse = py.memory_info().rss/(1024*1024*1024)
    memoryUse2 = py.memory_info()[0]/2.**30 # memory use in GB...I think
    print('memory use:', memoryUse, "GB")
    print('memory use:', memoryUse2, "GB")
    CPU_Pct=str(round(float(os.popen('''grep 'cpu ' /proc/stat | awk '{usage=($2+$4)*100/($2+$4+$5)} END {print usage }' ''').readline()),2))
    print("CPU Usage = " + CPU_Pct + "%")
    cpu = py.cpu_percent()
    print("CPU : ",cpu)

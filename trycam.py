import imutils
import datetime
from threading import Thread
import cv2
import argparse
from skimage.measure import  compare_ssim as ssim
import time
import numpy as np

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
		return (self._end -self._start).total_seconds()
	
	def fps(self):
		return self._Frames/self.elapsed()

class CAM:
	def __init__(self,src=0):
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
			(self.grab,self.frame)= self.stream.read()

	def read(self):
		return self.frame

	def stop(self):
		self.stopped = True

	def release(self):
		self.stream.release()

		

class Finder:
	def __init__(self,vs):
		self.vs = vs.start()
		self.template1 = cv2.imread("template1.png")
		self.template2 = cv2.imread("template2.png")
		self.template3 = cv2.imread("template3.png")
		self.timestart = datetime.datetime.now()
		self.s1 = 0
		self.frame = vs.read()
		self.delta = 0
		self.simvaga = 0
		self.naovaga = 0
		self.frameDelta = 0
		if(self.timestart.hour>17):
			self.template = cv2.cvtColor(self.template2,cv2.COLOR_BGR2GRAY)
			print("template2")
		elif(self.timestart.hour>9):
			self.template = cv2.cvtColor(self.template3,cv2.COLOR_BGR2GRAY)
			print("template3")
		else:
			self.template = cv2.cvtColor(self.template1,cv2.COLOR_BGR2GRAY)
			print("template1")
		
		self.clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
		self.templateclahe = self.clahe.apply(self.template)
		self.templateblur = cv2.GaussianBlur(self.templateclahe, (21, 21), 0)
		self.templateframe = self.templateblur
		self.img = self.templateblur
		self.stopped = False
	
	def start(self):
		y = Thread(target=self.check, args=())
		x = Thread(target=self.update, args=())
		y.start()
		x.start()

	def stop(self):
		self.stopped = True
		cv2.destroyAllWindows()
		self.vs.stop()


	def update(self):
		while True:
			if self.stopped:
				return
			self.frame = self.vs.read()
			self.img = self.frame[114:410,730:1150]
			self.img = cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
			self.img = self.clahe.apply(self.img)
			self.img = cv2.GaussianBlur(self.img, (21, 21), 0)
			self.s1 = ssim(self.img,self.templateframe)
			self.frameDelta = cv2.absdiff(self.img,self.templateframe)
			if(self.s1>0.8):
				self.templateframe = self.img
			cv2.imshow("img1",self.frame)
			x = cv2.waitKey(10) & 0xff
			if(x==27):
				self.stop()
				break
			if(x==32):
				print("VAMO PORRAAAAAAAAAAA")

	def check(self):
		while True:
			time.sleep(0.5)
			if self.stopped:
				return

			if(self.s1<0.5):
				
				threshold = cv2.threshold(self.frameDelta,25,255,cv2.THRESH_BINARY)[1]
				threshold = cv2.dilate(threshold, None, iterations=2)
			
				n = np.sum(threshold==255)  
				if(n>50000):
					if(self.simvaga == 0):
						print("Carro na vaga")
						self.simvaga = 1
						self.naovaga = 0

				else:
					self.simvaga = 0
					if(self.naovaga == 0):
						print("Sem carro")
						self.naovaga = 1
					
			else:
				if(self.naovaga == 0):
					print("sem carro")
					self.naovaga = 1
					self.simvaga = 0

	def read(self):
		return self.vs.frame

	def readFrame(self):
		self.frame = self.vs.read()
		self.img = self.frame[114:410,730:1150]
		self.img = cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
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

ap = argparse.ArgumentParser()
ap.add_argument("-d","--display",type=int, default = -1,help="mostrar fps")
args = vars(ap.parse_args())

# vs = CAM(src ="rtsp://user:1@192.168.2.6:554/cam/realmonitor?channel=1&subtype=0" ).start()
vs = CAM(src ="rtsp://user:1@192.168.2.6:554/cam/realmonitor?channel=1&subtype=0" )
# fps = FPS().start()

xs = Finder(vs).start()

# while True:
# 	frame1 = xs.read()
# 	frame2 = xs.showTemplateClahe()
# 	cv2.imshow("img1",frame1)
# 	cv2.imshow("img2",frame2)

# 	print(xs.returnssim())
# 	x = cv2.waitKey(10) & 0xff
# 	if(x==27):
# 		vs.stop()
# 		break
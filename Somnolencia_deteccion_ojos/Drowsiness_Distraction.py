from scipy.spatial import distance
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import argparse
import imutils
import time
import dlib
import cv2

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])	
    # compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = distance.euclidean(eye[0], eye[3])	
    # compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)	
    # return the eye aspect ratio
	return ear

def sound_alarm():
	# play an alarm sound
	playsound.playsound("C:\\Users\\c_j15\\OneDrive\\Documents\\Carlos\\Machine Learning\\Saturdays\\Proyecto\\Somnolencia y distractores\\Somnolencia 2\\Somnolencia_deteccion_ojos\\sound\\alarm2.mp3")


# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold for to set off the
# alarm
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 96
EYE_AR_NOT_DETECTED_FRAMES = 96

# initialize the frame counters for drowsiness and eyes distraction
# as well as a boolean used to indicate if the alarm is going off
COUNTER_DROWSINESS = 0
COUNTER_EYES_NOT_DETECTED = 0
ALARM_ON = False
ENVIO_ALERTA = False
MENSAJE_ALERTA = ""

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")# Dat file is the crux of the code

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start video capture
cap=cv2.VideoCapture(0)

while True:
	ret, frame=cap.read()
	frame = imutils.resize(frame, width=450)
	
	# detect faces in the grayscale frame
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	subjects = detect(gray, 0)
	ENVIO_ALERTA = False
	#Si subjects se encuentra vacio no se detectaron ojos en el cuadro 
	if not subjects:
		COUNTER_EYES_NOT_DETECTED += 1

		if COUNTER_EYES_NOT_DETECTED >= EYE_AR_NOT_DETECTED_FRAMES:
			# if the alarm is not on, turn it on			
			if not ALARM_ON and not ENVIO_ALERTA:
				ALARM_ON = True
				ENVIO_ALERTA = True
				MENSAJE_ALERTA = "***CONDUCTOR DISTRAIDO!***"
				#t = Thread(target=sound_alarm(frame,MENSAJE_ALERTA),)
				t = Thread(target=sound_alarm(),)
				t.deamon = True
				t.start()
				cv2.putText(frame, MENSAJE_ALERTA, (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)					
	else: 
		COUNTER_EYES_NOT_DETECTED = 0
		ALARM_ON = False
		
		
		for subject in subjects:
			# determine the facial landmarks for the face region, then
			# convert the facial landmark (x, y)-coordinates to a NumPy array
			shape = predict(gray, subject)
			shape = face_utils.shape_to_np(shape)#converting to NumPy Array

			# extract the left and right eye coordinates, then use the
			# coordinates to compute the eye aspect ratio for both eyes
			leftEye = shape[lStart:lEnd]
			rightEye = shape[rStart:rEnd]
			leftEAR = eye_aspect_ratio(leftEye)
			rightEAR = eye_aspect_ratio(rightEye)

			# average the eye aspect ratio together for both eyes
			ear = (leftEAR + rightEAR) / 2.0

			# compute the convex hull for the left and right eye
			leftEyeHull = cv2.convexHull(leftEye)
			rightEyeHull = cv2.convexHull(rightEye)
		
			# visualize each of the eyes
			cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
			cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		
			# check to see if the eye aspect ratio is below the blink
			# threshold, and if so, increment the blink frame counter
			if ear < EYE_AR_THRESH:
				#flag += 1
				COUNTER_DROWSINESS += 1

				# if the eyes were closed for a sufficient number of then sound the alarm
				if COUNTER_DROWSINESS >= EYE_AR_CONSEC_FRAMES:					
					# if the alarm is not on, turn it on
					if not ALARM_ON and not ENVIO_ALERTA:
						# check to see if an alarm file was supplied,
						# and if so, start a thread to have the alarm
						# sound played in the background
						ALARM_ON = True
						ENVIO_ALERTA = True
						MENSAJE_ALERTA = "***ALERTA DE SOMNOLENCIA!***"
						#t = Thread(target=sound_alarm(frame,MENSAJE_ALERTA),)
						t = Thread(target=sound_alarm(),)
						t.deamon = True
						t.start()
						cv2.putText(frame, MENSAJE_ALERTA, (10, 30),
							cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)						
										
		
			# otherwise, the eye aspect ratio is not below the blink
			# threshold, so reset the counter and alarm
			else:
				COUNTER_DROWSINESS = 0
				ALARM_ON = False
				
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
cv2.destroyAllWindows()
cap.release() 

import cv2
import numpy as np
import mediapipe as mp
import time
#import csv
from keras.models import model_from_json
from facelandMark import FaceMesh
#from datetime import datetime


emotion_dict = ["Marah", "Risih", "Takut", "Senyum", "Netral", "Sedih", "Terkejut"]
emotion_prediction = [[0, 0, 0, 0, 0, 0, 0]]
#emotion_dict = {0: "Marah", 1: "Senang", 2: "Netral", 3: "Takut"}

# load json and create model
json_file = open('emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into new modela
emotion_model.load_weights("emotion_model.h5")
print("Loaded model from disk")

faceMesh = FaceMesh(min_detection_confidence=0.7,min_tracking_confidence=0.7)
# start the webcam feed
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture ('D:\\Project Magang 2\\Emotion-Detection\\testing.mp4')
#cap = cv2.VideoCapture ('D:\\Project Magang 2\\Emotion-Detection\\videoface.mp4')
prev_frame_time = 0
new_frame_time = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    if not ret:
        break
    face_detector = cv2.CascadeClassifier('D:\\Project Magang 2\Emotion-Detection\haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #results = faceDetection.process(gray_frame)
    #print(results)

    # detect faces available on camera
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # take each face available on the camera and Preprocess it
    for (x, y, w, h) in num_faces:
        # cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # predict the emotions
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        cv2.putText(frame, f'{emotion_dict[maxindex]}', (x+5, y-20), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        

        
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    
    faceLandmarks = faceMesh.findFaceLandMarks(image=frame,draw=True)

    cv2.putText(frame, f'{time.strftime("%H:%M:%S %m/%d/%Y")}', (20, 40), font, 1, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.rectangle(frame, (60,500), (400,300), (128, 255, 255), -1)

    cv2.putText(frame, f'{emotion_dict[0]}', (60, 100), font, 1, (124, 252, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'{emotion_dict[1]}', (60, 150), font, 1, (124, 252, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'{emotion_dict[2]}', (60, 200), font, 1, (124, 252, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'{emotion_dict[3]}', (60, 250), font, 1, (124, 252, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'{emotion_dict[4]}', (60, 300), font, 1, (124, 252, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'{emotion_dict[5]}', (60, 350), font, 1, (124, 252, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'{emotion_dict[6]}', (60, 400), font, 1, (124, 252, 0), 2, cv2.LINE_AA)

    cv2.putText(frame, f': {round(emotion_prediction[0][0], 2)}', (200, 100), font, 1, (124, 252, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f': {round(emotion_prediction[0][1], 2)}', (200, 150), font, 1, (124, 252, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f': {round(emotion_prediction[0][2], 2)}', (200, 200), font, 1, (124, 252, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f': {round(emotion_prediction[0][3], 2)}', (200, 250), font, 1, (124, 252, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f': {round(emotion_prediction[0][4], 2)}', (200, 300), font, 1, (124, 252, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f': {round(emotion_prediction[0][5], 2)}', (200, 350), font, 1, (124, 252, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f': {round(emotion_prediction[0][6], 2)}', (200, 400), font, 1, (124, 252, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'FPS: {int(fps)}', (1100, 40), font, 1, (124, 252, 0), 2, cv2.LINE_AA)

    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) == ord('a'):
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import numpy as np
import time
from keras.models import model_from_json
from facelandMark import FaceMesh


emotion_dict = {0: "Marah", 1: "Risih", 2: "Takut", 3: "Senyum", 4: "Netral", 5: "Sedih", 6: "Terkejut"}
#emotion_dict = {0: "Marah", 1: "Senang", 2: "Netral", 3: "Takut"}

# load json and create model
json_file = open('emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into new model
emotion_model.load_weights("emotion_model.h5")
print("Loaded model from disk")

faceMesh = FaceMesh(min_detection_confidence=0.7,min_tracking_confidence=0.7)
# start the webcam feed
#cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture ('D:\\Project Magang 2\\Emotion-Detection\\testing.mp4')
cap = cv2.VideoCapture (0)
prev_frame_time = 0
new_frame_time = 0

while True:
    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    if not ret:
        break
    face_detector = cv2.CascadeClassifier('D:\\Project Magang 2\Emotion-Detection\haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces available on camera
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # take each face available on the camera and Preprocess it
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # predict the emotions
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), font, 1, (124, 252, 0), 2, cv2.LINE_AA)
        
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time

        fps = int(fps)
        fps = str(fps)

    faceLandmarks = faceMesh.findFaceLandMarks(image=frame,draw=True)
    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) == ord('a'):
        break

cap.release()
cv2.destroyAllWindows()
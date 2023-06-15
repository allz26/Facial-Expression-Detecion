from flask import Flask, render_template, request, Response
import cv2
from keras.models import model_from_json
import numpy as np
from PIL import Image
import time
import base64
import io
#from datetime import datetime

# import required modules
import mysql.connector
import json

# create connection object
con = mysql.connector.connect(
host="localhost", user="root",
password="", database="deteksi_emosi")

# create cursor object
cursor = con.cursor()
app = Flask(__name__)
emotion_dict = ["Marah", "Risih", "Takut", "Senyum", "Netral", "Sedih", "Terkejut"]
json_file = open('emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

face_detector = cv2.CascadeClassifier('D:\\Project Magang 2\Emotion-Detection\haarcascade_frontalface_default.xml')

# load weights into new model
emotion_model.load_weights("emotion_model.h5")
hasil_dict = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/after', methods=['POST'])
def after():
    global hasil_dict
    img = request.files['file']

    img = Image.open(img.stream)
    img = img.convert("RGB")
    img = np.asarray(img)

    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #results = faceDetection.process(gray_frame)
    #print(results)

    # detect faces available on camera
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
    # cv2.putText({time.strftime("%H:%M:%S %m/%d/%Y")})
    #now = datetime.now()

    # take each face available on the camera and Preprocess it
    for (x, y, w, h) in num_faces:
        
        #cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))

        cv2.putText(img, f'{emotion_dict[maxindex]}', (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)
        
        # predict the emotions
        hasil = emotion_dict[maxindex]
        hasil_dict.append({"Prediksi Emosi": hasil})
    hasil_dict = json.dumps({"waktu":time.strftime("%H:%M:%S %m/%d/%Y"), "hasil": hasil_dict})
    insert_query = "INSERT INTO data_emosi (json_data) VALUES (%s)"
    values = (hasil_dict,)
    cursor.execute(insert_query, values)
    con.commit()

    img= cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    _, encoded_img = cv2.imencode('.jpg', img)
    base64_img = base64.b64encode(encoded_img).decode('utf-8')
    data_uri = 'data:image/jpeg;base64,' + base64_img

    rendered_html = render_template('after.html', data_uri=data_uri)
    return Response(rendered_html, mimetype="text/html")

@app.route("/api", methods=["GET"])
def hasil_get():
    return hasil_dict
if __name__== "__main__":
    app.run(debug=True)
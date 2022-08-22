import cv2 as cv
import numpy as np
from flask import Flask, render_template, Response,request
from keras.models import load_model
import Util

app = Flask(__name__)
camera=cv.VideoCapture(0)

def generate_frames():
    # 加载人脸识别和情绪识别模型
    face_classifier = cv.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    classifier = load_model('EmotionDetectionModel.h5')
    class_labels = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    while True:
        ## read the camera frame
        success, frame = camera.read()
        # 识别图片，并标注情绪
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv.resize(roi_gray, (48, 48), interpolation=cv.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = np.array(roi)
                roi = np.expand_dims(roi, axis=0)

                preds = classifier.predict(roi)[0]
                label = class_labels[preds.argmax()]
                label_position = (x, y)
                cv.putText(frame, label, label_position, cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            else:
                cv.putText(frame, 'No Face Found', (20, 20), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

        if not success:
            break
        else:
            ret, buffer = cv.imencode('.jpg', frame)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route("/")
def index1():
    return render_template("index1.html")

@app.route("/pic_recognition", methods=['GET', 'POST'])
def pic_recognition():
    return render_template("pic_recognition.html")

@app.route("/video_recognition_cam")
def video_recognition_cam():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/video_recognition")
def video_recognition():
    return render_template("video_recognition.html")

@app.route("/after",methods=["POST","GET"])
def after():
    img = request.files['file1']
    img.save('static/file.jpg')
    ###############
    img1 = cv.imread('static/file.jpg')
    img_final, emo_freq = Util.draw_allround_faces_on_image(img1)
    emo_most_freq= Util.get_most_freq_emo(emo_freq)
    cv.imwrite('static/img_final.jpg', img_final)
    return render_template("after.html", data=emo_most_freq)
if __name__ == "__main__":
    app.run(debug=True)

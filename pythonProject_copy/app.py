import cv2 as cv
import numpy as np
from flask import Flask, render_template, Response,request

import Util

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index1.html")

@app.route("/pic_recognition", methods=['GET', 'POST'])
def pic_recognition():
    return render_template("pic_recognition.html")

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

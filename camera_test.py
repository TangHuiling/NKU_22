import cv2 as cv
import numpy as np
from keras.models import load_model


##############################
# 由于摄像头实时测试时调用Util包会出现卡顿、掉帧的情况
# 因此这里直接写测试代码，不调用工具包
##############################


face_classifier = cv.CascadeClassifier('pythonProject_copy/haarcascade_frontalface_alt2.xml')
classifier = load_model('pythonProject_copy/EmotionDetectionModel.h5')
class_labels = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
# 开启摄像头
cap = cv.VideoCapture(0, cv.CAP_DSHOW)
print('cameraIsOpened? ', cap.isOpened())

while True:
    # 读取图片
    ret, frame = cap.read()
    labels = []
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

    cv.imshow('Emotion Detector', frame)
    # 按q键退出
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()

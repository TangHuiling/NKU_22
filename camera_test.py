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
    # 图片灰度化
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # scaleFactor--表示在前后两次相继的扫描中，搜索窗口的比例系数。设置为1.3，即每次搜索窗口依次扩大30%
    # minNeighbors--表示构成检测目标的相邻矩形的最小个数(默认为3个)，此处设置为5
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # 在返回的人脸位置处画上方框
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        # ROI调整大小：参数interpolation表示“插值方式”，INTER_AREA表示使用像素区域关系进行重采样
        roi_gray = cv.resize(roi_gray, (48, 48), interpolation=cv.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            # 图像归一化
            roi = roi_gray.astype('float') / 255.0
            roi = np.array(roi)
            # roi拓展维度：(1, 48, 48)
            roi = np.expand_dims(roi, axis=0)
            # 使用情绪识别模型进行预测，得到样本属于每一个类别的概率
            preds = classifier.predict(roi)[0]
            # 获得预测结果中可能性最高的情绪标签
            label = class_labels[preds.argmax()]
            label_position = (x, y)
            # 将情绪标签以文字形式标注在图片相应人脸的位置（左上角）
            cv.putText(frame, label, label_position, cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            # 图像纯黑的情况
            cv.putText(frame, 'No Face Found', (20, 20), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    cv.imshow('Emotion Detector', frame)
    # 按q键退出
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()

import cv2
import numpy as np
from  keras.models import model_from_json
from keras.preprocessing import image

#加载模型

model=model_from_json('fer.json','r').read()

#加载权重文件

model.load.weights('fer.h5')

#加载haa的模型
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#打开摄像头

cap=cv2.VideoCapture(0)

while True:
    ret,frame=cap.read()
    if not ret:
        break
   #转成灰度图
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.32,2)

    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        roi_gray=gray[y:y+h,x:x+w]
        roi_gray=cv2.resize(roi_gray,(48,48))#图片变成48*48

        #将图片转换为数组
        #感兴趣的人脸区域转成灰度
        pixels=image.img_to_array(roi_gray)
        #改变图片的维度
        pixels=np.expand_dims(pixels,axis=0)
        #归一化处理
        pixels/=255


        pred=model.predict(pixels)

        max_index=np.argmax(pred[0])

        emotions=('angry','disgust','fear','happy','sad','surprise','neutral')
        pred_emotions=emotions[max_index]

        cv2.putText(frame,pred_emotions,(int(x),int(y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2))

    resize_frame=cv2.resize(frame,(1000,700))
    cv2.imshow('frame',resize_frame)
    if cv2.waitKey(1)==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import cv2 as cv
import numpy as np
from keras.models import load_model


############################################
# 工具类
# func 1: get_faces_from_srcImage(srcImage)
#           【功能】给出图片，返回原图中的人脸区域
# func 2: get_emotion_from_roi(roi_gray)
#           【功能】将处理好的48*48的灰度面部图交给训练好的情绪分类器处理，得到情绪类别
# func 3: draw_faces_on_image(srcImage, faces)
#           【功能】在原图上框出得到的脸部区域，并标注其情绪种类
# func 4: rotate_bound(image, angle)
#           【功能】按一定角度顺时针旋转图片，并保证图片完整，不被裁剪
# func 5: draw_allround_faces_on_image(image)
#           【功能】360度旋转图片，每30度识别一次+画一次
############################################


# description: 给出图片，返回原图中的人脸区域
# params: 图片
# returns: 人脸区域
def get_faces_from_srcImage(srcImage):
    # 将原图转化为灰度图
    grayImage = cv.cvtColor(srcImage, cv.COLOR_BGR2GRAY)
    # 得到OpenCV人脸识别的分类器
    face_classifier = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    # 检测人脸
    faces = face_classifier.detectMultiScale(grayImage, scaleFactor=1.3, minNeighbors=5)
    return faces


# description: 将处理好的48*48的灰度面部图交给训练好的情绪分类器处理，得到情绪类别
# params: 48*48的灰度图
# returns: 情绪对应的名称
def get_emotion_from_roi(roi_gray):
    # 加载情绪分类器
    classifier = load_model('../Model/EmotionDetectionModel.h5')
    # 给出情绪标签对应的文字
    class_labels = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    # 若roi_gray全0，即全黑，则表示该区域中无脸部图像
    if np.sum([roi_gray]) != 0:
        roi = roi_gray.astype('float') / 255.0
        roi = np.array(roi)
        roi = np.expand_dims(roi, axis=0)
        # 开始预测
        prediction = classifier.predict(roi)[0]
        label = class_labels[prediction.argmax()]
        return label
    else:
        return 'No Face Found'


# description: 在原图上框出得到的脸部区域，并标注其情绪种类
# params: 图片，脸部区域，情绪识别分类器
# returns: 被注明的图片
def draw_faces_on_image(srcImage, faces):
    if len(faces) == 0:
        print('No faces!')
    for (x, y, w, h) in faces:
        # 用cv画出矩形范围
        cv.rectangle(srcImage, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=2)
        # 将该范围作为roi区域，进行情绪识别
        roi = srcImage[y:y + h, x:x + w]
        # 转换为灰度图
        roi_gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
        # 统一大小
        roi_gray = cv.resize(roi_gray, (48, 48), interpolation=cv.INTER_AREA)
        # 使用情绪分类器进行识别，并将得到的情绪名称标注在图片上
        label = get_emotion_from_roi(roi_gray)
        label_position = (x, y)
        cv.putText(srcImage, label, label_position, cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0),
                   thickness=2)


# description: 按一定角度旋转图片，并保证图片完整，不被裁剪
# params: 图片，需要旋转的角度
# returns: 旋转后的图片，旋转矩阵
def rotate_bound(image, src_h, src_w, angle, i):
    # 获得图片的高、宽，中心坐标
    actual_h, actual_w = image.shape[:2]
    (center_x, center_y) = (actual_w // 2, actual_h // 2)
    if i == 0:
        return image
    else:
        # 得到旋转矩阵（为使其顺时针选择，将角度设置为负），缩放因子=1.0
        rotationMatrix = cv.getRotationMatrix2D(center=(center_x, center_y), angle=-angle, scale=1.0)
        # 用于计算实际画布大小的旋转矩阵：旋转角度是对于原图而言的
        fake_M = cv.getRotationMatrix2D(center=(center_x, center_y), angle=-angle * i, scale=1.0)
        cos = np.abs(fake_M[0, 0])
        sin = np.abs(fake_M[0, 1])
        # 为保证选择后的图片完整，需要扩大画布，新画布的维度值如下
        newWidth = int((src_h * sin) + (src_w * cos))
        newHeight = int((src_h * cos) + (src_w * sin))
        # 平移图像→平移中心点
        rotationMatrix[0, 2] += (newWidth / 2) - center_x
        rotationMatrix[1, 2] += (newHeight / 2) - center_y

        rotated_image = cv.warpAffine(image, rotationMatrix, (newWidth, newHeight), borderValue=(0, 0, 0))
        return rotated_image


# description: 360度旋转图片，每30度识别一次+画一次
# params: 图片
# returns: 360度识别人脸并标注情绪
def draw_allround_faces_on_image(image):
    # 旋转角度和总共旋转的次数
    ANGLE = 45
    TIMES = 360 // ANGLE
    # 原图的宽高
    (IMG_H, IMG_W) = image.shape[:2]
    # 已标注的面部列表
    drawn_faces = []
    # 同一张图片中可能出现的面部面积比值
    min_faces_ratio = 0.5
    max_faces_ratio = 2
    for i in np.arange(TIMES):
        image = rotate_bound(image, IMG_H, IMG_W, ANGLE, i)
        # 从图片中获取人脸区域
        faces = get_faces_from_srcImage(image)
        # 检测所识别到的人脸范围内是否面积大小相差过大，将不合理的去除
        right_faces = []
        for face in faces:
            (x, y, w, h) = face
            is_right = True
            area = np.float(w * h)
            for drawn_face in drawn_faces:
                (dx, dy, dw, dh) = drawn_face
                darea = np.float(dw * dh)
                if area/darea > max_faces_ratio or area/darea < min_faces_ratio:
                    is_right = False
                    break
            if is_right:
                right_faces.append(face)
                drawn_faces.append(face)
        # 在原图上框出得到的脸部区域，并标注其情绪种类
        draw_faces_on_image(image, right_faces)
        # # 展示旋转的过程
        # cv.imshow('rotated image:', image)
        # cv.waitKey()
    # 最后旋转一次，摆正原图q
    image = rotate_bound(image, IMG_H, IMG_W, ANGLE, TIMES)
    return image

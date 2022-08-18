import cv2 as cv
import numpy as np
from keras.models import load_model
import face_recognition


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
#           【功能】360度旋转图片，每旋转一定角度识别一次+画一次
# func 6: classify_pHash(image1, image2)
#           【功能】 感知哈希算法，描述两张图片的相似度 (效果不佳)
############################################


# description: 给出图片，返回原图中的人脸区域
# params: 图片
# returns: 人脸区域 (一个元素为(top, right, bottom, left)元组的list)
def get_faces_from_image(image):
    # 模型选择一：用OpenCV人脸识别分类器
    # 将原图转化为灰度图
    # gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # # 得到OpenCV人脸识别的分类器
    # face_classifier = cv.CascadeClassifier('pythonProject_copy/haarcascade_frontalface_default.xml')
    # # 检测人脸
    # faces = face_classifier.detectMultiScale(image, scaleFactor=1.3, minNeighbors=5)

    # 模型选择二：使用face_recognition模块
    face_locations = face_recognition.face_locations(image)
    return face_locations


# description: 将处理好的48*48的灰度面部图交给训练好的情绪分类器处理，得到情绪类别
# params: 48*48的灰度图
# returns: 情绪对应的名称
def get_emotion_from_roi(roi_gray):
    # 加载情绪分类器
    classifier = load_model('pythonProject_copy/EmotionDetectionModel.h5')
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
# params: 图片，脸部区域，统计的情绪频率
# returns: 无
def draw_faces_on_image(image, faces, emo_freq):
    if len(faces) == 0:
        print('Draw No faces!')
    for (top, right, bottom, left) in faces:
        # 用cv画出矩形范围
        cv.rectangle(image, (left, top), (right, bottom), color=(255, 0, 0), thickness=2)
        # 将该范围作为roi区域，进行情绪识别
        roi = image[top:bottom, left:right]
        # 转换为灰度图
        roi_gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
        # 统一大小
        roi_gray = cv.resize(roi_gray, (48, 48), interpolation=cv.INTER_AREA)
        # 使用情绪分类器进行识别，并将得到的情绪名称标注在图片上
        label = get_emotion_from_roi(roi_gray)
        label_position = (left, top)
        cv.putText(image, label, label_position, cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0),
                   thickness=2)
        # 对应情绪计数+1
        emo_freq[label] += 1


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
        rotation_matrix = cv.getRotationMatrix2D(center=(center_x, center_y), angle=-angle, scale=1.0)
        # 用于计算实际画布大小的旋转矩阵：旋转角度是对于原图而言的
        fake_matrix = cv.getRotationMatrix2D(center=(center_x, center_y), angle=-angle * i, scale=1.0)
        cos = np.abs(fake_matrix[0, 0])
        sin = np.abs(fake_matrix[0, 1])
        # 为保证选择后的图片完整，需要扩大画布，新画布的维度值如下
        new_width = int((src_h * sin) + (src_w * cos))
        new_height = int((src_h * cos) + (src_w * sin))
        # 平移图像→平移中心点
        rotation_matrix[0, 2] += (new_width / 2) - center_x
        rotation_matrix[1, 2] += (new_height / 2) - center_y

        rotated_image = cv.warpAffine(image, rotation_matrix, (new_width, new_height), borderValue=(0, 0, 0))
        return rotated_image


# description: 360度旋转图片，每30度识别一次+画一次
# params: 图片
# returns: 360度识别人脸并标注情绪的标注图，以及出现情绪频率的dict
def draw_allround_faces_on_image(image):
    emo_freq = {
        'Angry': 0, 'Fear': 0, 'Happy': 0, 'Neutral': 0, 'Sad': 0, 'Surprise': 0
    }
    # 旋转角度和总共旋转的次数
    ANGLE = 45
    TIMES = 360 // ANGLE
    # 原图的宽高
    (IMG_H, IMG_W) = image.shape[:2]
    # 已标注的面部列表
    drawn_faces_encoding = []
    for i in np.arange(TIMES):
        i_right_faces = []
        image = rotate_bound(image, IMG_H, IMG_W, ANGLE, i)
        i_faces_loc = get_faces_from_image(image)
        for (top, right, bottom, left) in i_faces_loc:
            # 比较图片相似度
            face_img_encoding = face_recognition.face_encodings(image[top:bottom, left:right, :], known_face_locations=[
                (0, right - left, bottom - top, 0)])
            if len(face_img_encoding) != 0:
                is_same_face = face_recognition.compare_faces(drawn_faces_encoding, face_img_encoding[0], tolerance=0.5)
                # 若与已标注的人脸集drawn_faces_encoding中某张人脸相同，则不再标注当前的face_img，否则重复
                if ~np.array(is_same_face).any():
                    i_right_faces.append((top, right, bottom, left))
                    drawn_faces_encoding.append(face_img_encoding[0])
        draw_faces_on_image(image, i_right_faces, emo_freq)
        # # 展示旋转的过程
        # cv.imshow('rotated image:', image)
        # cv.waitKey()
    # 最后旋转一次，摆正原图q
    image = rotate_bound(image, IMG_H, IMG_W, ANGLE, TIMES)
    return image, emo_freq


# desciption: 感知哈希算法，描述两张图片的相似度 (效果不好)
# params: 两张图片
# return: 两张图片的汉明距离
def classify_pHash(image1, image2):
    image1 = cv.resize(image1, (32, 32))
    image2 = cv.resize(image2, (32, 32))
    gray1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
    # 将灰度图转为浮点型，再进行dct变换
    dct1 = cv.dct(np.float32(gray1))
    dct2 = cv.dct(np.float32(gray2))
    # 取左上角的8*8，这些代表图片的最低频率
    # 这个操作等价于c++中利用opencv实现的掩码操作
    # 在python中进行掩码操作，可以直接这样取出图像矩阵的某一部分
    dct1_roi = dct1[0:8, 0:8]
    dct2_roi = dct2[0:8, 0:8]
    hash1 = getHash(dct1_roi)
    hash2 = getHash(dct2_roi)
    return Hamming_distance(hash1, hash2)


# 输入灰度图，返回hash
def getHash(image):
    average = np.mean(image)
    hash = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] > average:
                hash.append(1)
            else:
                hash.append(0)
    return hash


# 计算汉明距离
def Hamming_distance(hash1, hash2):
    num = 0
    for index in range(len(hash1)):
        if hash1[index] != hash2[index]:
            num += 1
    return num

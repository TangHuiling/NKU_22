from keras.models import load_model
import cv2 as cv
import numpy as np
import face_recognition
import Util

#####################
# 用户上传图片得到情绪检测结果
# 使用的人脸识别模型是第三方模块包 face_recognition
# 网站：https://github.com/ageitgey/face_recognition
# 安装需要下载：
# pip install boost
# pip install cmake
# pip install dlib
# pip install face-recognition
#####################


# description: 给出图片，返回原图中的人脸区域
# params: 图片
# returns: 人脸区域
def emotion_detect_on_image(src_image):
    # 在原图上框出得到的脸部区域，并标注其情绪种类
    image, emo_freq = Util.draw_allround_faces_on_image(src_image)
    # 显示图片
    cv.imshow('Emotion Detector', image)
    cv.waitKey()
    # 输出6中情绪出现的频率
    print(emo_freq)
    print(Util.get_most_freq_emo(emo_freq))


if __name__ == '__main__':
    # 纯黑测试
    # black_img = np.zeros((48, 48, 3), np.uint8)
    # emotion_detect_on_image(black_img)

    # 传入图片识别情绪
    image_filename = 'kids.jpg'
    image = cv.imread(image_filename)
    emotion_detect_on_image(image)
    cv.destroyAllWindows()

    # # 人脸识别定位测试
    # image_filename = 'images/with_angle/kids.jpg'
    # image = face_recognition.load_image_file(image_filename)
    # face_location = face_recognition.face_locations(image)
    # # top, right, bottom, left
    # for (top, right, bottom, left) in face_location:
    #     print((top, right, bottom, left))
    #     cv.imshow('face loc:', image[top:bottom, left:right, :])
    #     cv.waitKey()

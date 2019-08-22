import matplotlib.pyplot as plt
import os

from face.rasp_face_recognition import camera
from face.rasp_face_recognition.dataset import extract_face
from face.rasp_face_recognition.face_recognition import train_model, recognize_pic


def help():
    print("****************************")
    print("***                    ")
    print("***   1 - 拍照          ")
    print("***   2 - 提取人脸区域   ")
    print("***   3 - 识别人脸       ")
    print("***   x - 退出           ")
    print("***                      ")
    print("****************************")


def get_face(pic_file):
    face = extract_face(pic_file)
    plt.imshow(face)
    plt.show()


if __name__ == '__main__':

    if not os.path.exists("svm_classifier.model"):
        train_model()

    help()

    pic_file = '/home/pi/Desktop/image.jpg'

    while True:
        ch = input("Input instruction code：")
        if ch == '1':
            print("start taking photo...")
            if camera.take_picture(pic_file):
                print("finish!")
            else:
                print("fail")
                continue
        elif ch == '2':
            print("start extracting face...")
            get_face(pic_file)
        elif ch == '3':
            print("start recognizing face...")
            recognize_pic(pic_file)
        elif ch == 'x':
            break
        else:
            help()




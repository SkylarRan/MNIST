import os
import time

from PIL import Image
from numpy import expand_dims
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.svm import SVC
import matplotlib.pyplot as plt

from face.rasp_face_recognition.dataset import extract_face, img_to_encoding, create_model, Dataset


class SVMModel:
    def __init__(self):
        self.model = None
        self.out_encoder = None

    def build_model(self, dataset):
        self.model = SVC(kernel='linear', probability=True)
        # 需要将每个名人姓名的字符串目标变量转换为整数。
        self.out_encoder = LabelEncoder()
        self.out_encoder.fit(dataset.y_train)

    def train(self, dataset):
        # 首先，对人脸嵌入向量进行归一化是一种很好的做法。这是一个很好的实践，因为向量通常使用距离度量进行比较。
        in_encoder = Normalizer(norm='l2')
        X_train = in_encoder.transform(dataset.X_train)

        y_train = self.out_encoder.transform(dataset.y_train)
        self.model.fit(X_train, y_train)

        # 使用拟合模型对训练和测试数据集中的每个样本进行预测，然后计算分类正确率来实现。
        # predict
        yhat_train = self.model.predict(X_train)
        # score
        score_train = accuracy_score(y_train, yhat_train)
        # summarize
        print('Accuracy: train=%.3f,' % (score_train * 100))

    def save_model(self, file_path="svm_classifier.model"):
        joblib.dump((self.model, self.out_encoder), file_path)

    def load_model(self, file_path="svm_classifier.model"):
        self.model, self.out_encoder = joblib.load(file_path)

    def predict(self, image, model):
        image = extract_face(image)
        image = expand_dims(image, axis=0)
        image_embedding = img_to_encoding(image, model)
        pre_class = self.model.predict(image_embedding)
        probability = self.model.predict_proba(image_embedding)
        print("pre_class: {}".format(pre_class))
        print(self.out_encoder.classes_)
        print("probability: {}".format(probability))
        class_index = pre_class[0]
        probability = probability[0, class_index] * 100
        pre_name = self.out_encoder.inverse_transform(pre_class)
        print('Predicted: %s (%.3f)' % (pre_name[0], probability))
        return pre_name[0], probability


def train_model():
    facenet = create_model()
    print("加载数据")
    dataset = Dataset("face-dataset/train/")
    dataset.load(facenet)
    print("训练数据集 ---- 图片：{}， 标签：{}".format(dataset.X_train.shape, dataset.y_train.shape))

    svm = SVMModel()
    svm.build_model(dataset)
    svm.train(dataset)
    svm.save_model("svm_classifier.model")
    print("save model successfully!")


def recognize_pic(pic_file):
        # 使用保存的模型
        facenet = create_model()
        svm = SVMModel()
        svm.load_model("svm_classifier.model")
        print(svm.model, svm.out_encoder)
        print("load model successfully!")

        # pre_name, probability = svm.predict(pic_file, facenet)
        # plt.imshow(Image.open(pic_file))
        # title = '%s (%.3f)' % (pre_name, probability)
        # plt.title(title)
        # plt.show()


# train_model()
recognize_pic("test1.jpg")


# filePath = "face-dataset/train/bill"
# timeArray = time.localtime(os.path.getmtime(filePath))
# otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
# print(otherStyleTime)
# timeArray = time.localtime(os.path.getctime(filePath))
# otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
# print(otherStyleTime)

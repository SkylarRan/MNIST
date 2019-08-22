from os import listdir
from os.path import isdir

from PIL import Image
from mtcnn.mtcnn import MTCNN
from numpy import asarray, around
from scipy import linalg
from tensorflow import keras


def extract_face(filename, required_size=(160, 160)):
    """
    从图片文件中加载照片，并返回提取的人脸。
    :param filename: 图片文件
    :param required_size:人脸区域图片的尺寸
    :return: 人脸区域的数组形式
    """
    image = Image.open(filename)
    image = image.convert('RGB')
    pixels = asarray(image)

    # 创建一个 MTCNN 人脸检测器类，并使用它来检测加载的照片中所有的人脸。
    detector = MTCNN()

    # 结果是一个边界框列表，其中每个边界框定义了边界框的左下角，以及宽度和高度。
    results = detector.detect_faces(pixels)

    # 确定边界框的像素坐标,有时候库会返回负像素索引，可以通过取坐标的绝对值来解决这一问题。
    x1, y1, width, height = results[0].get("box")
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height

    # 使用这些坐标来提取人脸
    face = pixels[y1:y2, x1:x2]

    # 将这个人脸的小图像调整为所需的尺寸；
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array


def img_to_encoding(images, model):
    # Color image loaded by OpenCV is in BGR mode. But Matplotlib displays in RGB mode.
    # 这里的操作实际是对channel这一dim进行reverse，从BGR转换为RGB
    images = images[..., ::-1]
    images = around(images / 255.0, decimals=12)  # np.around是四舍五入，其中decimals是保留的小数位数,这里进行了归一化
    # https://stackoverflow.com/questions/44972565/what-is-the-difference-between-the-predict-and-predict-on-batch-methods-of-a-ker
    if images.shape[0] > 1:
        embedding = model.predict(images, batch_size=128)  # predict是对多个batch进行预测，这里的128是尝试后得出的内存能承受的最大值
    else:
        embedding = model.predict_on_batch(images)  # predict_on_batch是对单个batch进行预测
    # 报错，operands could not be broadcast together with shapes (2249,128) (2249,)，因此要加上keepdims = True
    embedding = embedding / linalg.norm(embedding, axis=1,
                                        keepdims=True)  # 注意这个项目里用的keras实现的facenet模型没有l2_norm，因此要在这里加上

    return embedding


class Dataset:
    def __init__(self, path_name):
        # 训练集
        self.X_train = None
        self.y_train = None
        # 数据集加载路径
        self.path_name = path_name

    def load_faces(self, directory, per_num):
        """
        提取同一目录下的所有人脸
        :param directory: 如“face-dataset/train/skylar”，该目录下为skylar的个人照片
        :param per_num: 提取的每人的照片数
        :return:
        """
        faces = list()
        for filename in listdir(directory):
            path = directory + filename
            face = extract_face(path)
            faces.append(face)
            if len(faces) >= per_num:
                break

        return faces

    def load_dataset(self, directory, per_num):
        """
        提取数据集中的所有人脸数据，为每个检测到的人脸分配标签
        :param directory: 如 “face-dataset/train/”
        :return:
        """
        X, y = list(), list()
        for subdir in listdir(directory):
            path = directory + subdir + '/'
            # skip any files that might be in the dir
            if not isdir(path):
                continue
            # load all faces in the subdirectory
            faces = self.load_faces(path, per_num)
            # create labels
            labels = [subdir for _ in range(len(faces))]
            # summarize progress
            print('>loaded %d examples for class: %s' % (len(faces), subdir))
            # store
            X.extend(faces)
            y.extend(labels)
        return asarray(X), asarray(y)

    # 加载数据集
    def load(self, model, per_num=3):
        """

        :param model:
        :param per_num:
        :return:
        """
        # 加载数据集到内存
        images, labels = self.load_dataset(self.path_name, per_num)
        # 生成128维特征向量
        X_embedding = img_to_encoding(images, model)  # 考虑这里分批执行，否则可能内存不够，这里在img_to_encoding函数里通过predict的batch_size参数实现
        # 输出训练集、验证集和测试集的数量
        print('X_train shape', X_embedding.shape)
        print('y_train shape', labels.shape)
        print(X_embedding.shape[0], 'train samples')
        # 这里对X_train就不再进一步normalization了，因为已经在facenet里有了l2_norm
        self.X_train = X_embedding
        self.y_train = labels


def create_model():
    model = keras.models.load_model('facenet_keras.h5')
    print('Loaded Model, inputs:{}, outputs:{}'.format(model.inputs, model.outputs))
    return model
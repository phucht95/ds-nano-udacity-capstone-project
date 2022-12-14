import numpy as np
from keras.backend import set_session, get_session
from keras.layers import Dropout, GlobalAveragePooling2D, Dense
from keras.models import model_from_json, Sequential, Model, load_model
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, \
Flatten, Dense, Lambda, ELU, Cropping2D, Dropout, BatchNormalization
from keras.layers.convolutional import Convolution2D, Conv2D
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image
import tensorflow as tf
import json
import cv2


class PredictEngine:
    def __init__(self, sess=tf.Session(), graph=tf.get_default_graph(),\
            bottleneck_feature_path='bottleneck_features/DogResnet50Data.npz', \
            label_name_path="dog_names.json", weight='saved_models/weights.best.Resnet50.hdf5', \
            cascade='haarcascades/haarcascade_frontalface_alt.xml'):
        self.sess = sess
        self.graph = graph
        self.bottleneck_feature_path = bottleneck_feature_path
        self.label_name_path = label_name_path
        self.weight = weight
        self.cascade = cascade
        self.dog_names = self.load_dog_names()
        self.model = self.init_model()

    def init_model(self):
        with self.graph.as_default():
            set_session(self.sess)
            bottleneck_features = np.load(self.bottleneck_feature_path)
            train_resnet_50 = bottleneck_features['train']
            Resnet_model = Sequential()
            Resnet_model.add(GlobalAveragePooling2D(input_shape=train_resnet_50.shape[1:]))
            Resnet_model.add(Dense(500, activation='relu'))
            Resnet_model.add(Dropout(0.4))
            Resnet_model.add(Dense(133, activation='softmax'))
            Resnet_model.load_weights(self.weight)
            self.graph = tf.get_default_graph()
            Resnet_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            self.graph = tf.get_default_graph()
            model = Resnet_model
            self.graph = tf.get_default_graph()
        return model

    def path_to_tensor(self, img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        return np.expand_dims(x, axis=0)

    def resnet_50_model(self):
        set_session(self.sess)
        resnet_model = ResNet50(weights='imagenet')
        self.graph = tf.get_default_graph()
        return resnet_model

    def resnet_50_model_shape(self):
        set_session(self.sess)
        resnet_shape_model = ResNet50(weights='imagenet', include_top=False)
        self.graph = tf.get_default_graph()
        return resnet_shape_model

    def ResNet50_predict_labels(self, img_path):
        with self.graph.as_default():
            img = preprocess_input(self.path_to_tensor(img_path))
        return np.argmax(self.resnet_50_model().predict(img))

    def dog_detector(self, img_path):
        prediction = self.ResNet50_predict_labels(img_path)
        return ((prediction <= 268) & (prediction >= 151))

    def face_detector(self, img_path):
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(self.cascade)
        faces = face_cascade.detectMultiScale(gray)
        return len(faces) > 0

    def load_dog_names(self):
        dog_names=[]
        with open(self.label_name_path) as json_file:
            dog_names = json.load(json_file)
        return dog_names
    
    def extract_Resnet50(self, tensor):
        return ResNet50(weights='imagenet', include_top=False).predict(preprocess_input(tensor))

    def resnet_predict_dog_breed(self, img_path):
        bottleneck_feature = self.extract_Resnet50(self.path_to_tensor(img_path))
        predicted = self.model.predict(bottleneck_feature)
        return self.dog_names[np.argmax(predicted)]

    def predict(self, img_path):
        with self.graph.as_default():
            set_session(self.sess)
            breed = self.resnet_predict_dog_breed(img_path).split(".")[-1]
            if self.dog_detector(img_path):
                return "This's maybe a dog and it's breed is {}".format(breed)
                
            elif self.face_detector(img_path) > 0:
                return "This's maybe a human and if it's a dog then it's belong to {}".format(breed)
                
            else:
                return "Unknown"

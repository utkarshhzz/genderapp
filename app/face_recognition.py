import numpy as np
import pandas as pd
from sklearn import metrics
import sklearn
import pickle
import matplotlib.pyplot as plt
import cv2

#load all models
haar=cv2.CascadeClassifier('F:/Face_Recognition/data preprocessuing/model/haarcascade_frontalface_default.xml') #cascade classifier
model_svm=pickle.load(open('F:/Face_Recognition/data preprocessing2/model_svm.pickle','rb')) #SVM model machine learning
pca_models=pickle.load(open('F:/Face_Recognition/data preprocessing2/pca_dict.pickle','rb')) #PCA model
model_pca=pca_models['pca']
mean_face_arr=pca_models['mean_face']


def faceRecognitionPipeline(filename,path=True):
    # read image
    if path:
        img = cv2.imread(filename)
    else:
        img = filename #array
    # convert into gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # crop the face (using haar cascade classifier)
    faces = haar.detectMultiScale(gray, 1.5, 3)
    predictions = []
    for x, y, w, h in faces:
        # cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)
        roi = gray[y:y+h, x:x+w]
        # normalization (0-1)
        roi = roi / 255.0
        # resize images (100, 100)
        if roi.shape[1] > 100:
            roi_resize = cv2.resize(roi, (100, 100), cv2.INTER_AREA)
        else:
            roi_resize = cv2.resize(roi, (100, 100), cv2.INTER_CUBIC)
        # Flattening
        roi_reshape = roi_resize.reshape(1, 10000)
        # subtract with mean
        roi_mean = roi_reshape - mean_face_arr
        # get eigen image apply roi_mean to pca model
        eigen_image = model_pca.transform(roi_mean)
        # Eigen Image for Visualization
        eig_img = model_pca.inverse_transform(eigen_image)
        # pass to ML model (svm) and get predictions
        results = model_svm.predict(eigen_image)
        prob_score = model_svm.predict_proba(eigen_image)
        prob_score_max = prob_score.max()
        # generate report
        text = "%s :%d" % (results[0], prob_score_max * 100)
        print(text)
        # defining color male blue female pink
        if results[0] == 'male':
            color = (255, 255, 0)
        else:
            color = (255, 0, 255)
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        cv2.rectangle(img, (x, y-30), (x+w, y), color, -1)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)
        output = {
            'roi': roi,
            'eig_img': eig_img,
            'prediction_name': results[0],
            'prob_score': prob_score_max,
        }
        predictions.append(output)
    return img, predictions
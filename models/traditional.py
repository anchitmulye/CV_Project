import cv2
import numpy as np
import joblib

kmeans_orb = joblib.load('saved_model/orb_kmeans.pkl')
svm_orb = joblib.load('saved_model/orb_svm.pkl')
kmeans_sift = joblib.load('saved_model/sift_kmeans.pkl')
svm_sift = joblib.load('saved_model/sift_svm.pkl')
orb = cv2.ORB_create(nfeatures=1000)
sift = cv2.SIFT_create()
categories = ['normal', 'potholes']


def predict_traditional_orb(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, descriptors = orb.detectAndCompute(gray, None)

    if descriptors is not None:
        words = kmeans_orb.predict(descriptors)
        hist, _ = np.histogram(words, bins=np.arange(kmeans_orb.n_clusters + 1))
    else:
        hist = np.zeros(kmeans_orb.n_clusters)

    prediction = svm_orb.predict([hist])[0]
    prob = svm_orb.predict_proba([hist])[0]

    category = categories[prediction]
    confidence = prob[prediction]

    return category, confidence


def predict_traditional_sift(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, descriptors = sift.detectAndCompute(gray, None)

    if descriptors is not None:
        words = kmeans_sift.predict(descriptors)
        hist, _ = np.histogram(words, bins=np.arange(kmeans_sift.n_clusters + 1))
    else:
        hist = np.zeros(kmeans_sift.n_clusters)

    prediction = svm_sift.predict([hist])[0]
    prob = svm_sift.predict_proba([hist])[0]

    category = categories[prediction]
    confidence = prob[prediction]

    return category, confidence

import cv2
import numpy as np
import joblib

# Load models once
kmeans = joblib.load('saved_model/kmeans.pkl')
svm = joblib.load('saved_model/svm.pkl')
orb = cv2.ORB_create(nfeatures=1000)

categories = ['normal', 'potholes']

def predict(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, descriptors = orb.detectAndCompute(gray, None)

    if descriptors is not None:
        words = kmeans.predict(descriptors)
        hist, _ = np.histogram(words, bins=np.arange(kmeans.n_clusters + 1))
    else:
        hist = np.zeros(kmeans.n_clusters)

    prediction = svm.predict([hist])[0]
    prob = svm.predict_proba([hist])[0]

    category = categories[prediction]
    return category, prob[prediction]

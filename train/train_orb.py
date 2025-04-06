import cv2
import os
import numpy as np
from sklearn.svm import SVC
import joblib
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

data_dir = 'Dataset'
categories = ['normal', 'potholes']

# ORB extractor
orb = cv2.ORB_create(nfeatures=1000)

# Store all descriptors
all_descriptors = []
labels = []
image_paths = []

print("Extracting features...")
for label, category in enumerate(categories):
    category_dir = os.path.join(data_dir, category)
    for image_name in tqdm(os.listdir(category_dir)):
        image_path = os.path.join(category_dir, image_name)
        image = cv2.imread(image_path)
        if image is None:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        if descriptors is not None:
            all_descriptors.extend(descriptors)
            image_paths.append(image_path)
            labels.append(label)

# Convert to numpy
all_descriptors = np.array(all_descriptors)

# Build vocabulary
print("Building visual vocabulary...")
kmeans = MiniBatchKMeans(n_clusters=1000, random_state=42)
kmeans.fit(all_descriptors)


# Build histograms (features)
def extract_bow_histogram(image_path, orb, kmeans):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, descriptors = orb.detectAndCompute(gray, None)
    if descriptors is not None:
        words = kmeans.predict(descriptors)
        hist, _ = np.histogram(words, bins=np.arange(kmeans.n_clusters + 1))
    else:
        hist = np.zeros(kmeans.n_clusters)
    return hist


print("Building feature vectors for each image...")
features = []
for path in tqdm(image_paths):
    hist = extract_bow_histogram(path, orb, kmeans)
    features.append(hist)

features = np.array(features)

# Train SVM
print("Training SVM classifier...")
svm = SVC(kernel='rbf', class_weight='balanced', probability=True)
svm.fit(features, labels)

# Save models
os.makedirs('saved_model', exist_ok=True)
joblib.dump(kmeans, 'saved_model/orb_kmeans.pkl')
joblib.dump(svm, 'saved_model/orb_svm.pkl')
print("Models saved!")

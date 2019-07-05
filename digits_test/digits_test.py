from skimage.io import imread
import numpy as np

# Loads data from the images into arrays and then generates an array with them
images = np.array([imread('1.jpg', as_gray=True), imread('1_alt.jpg', as_gray=True),
                   imread('2.jpg', as_gray=True), imread('2_alt.jpg', as_gray=True),
                   imread('3.jpg', as_gray=True), imread('3_alt.jpg', as_gray=True),
                   imread('4.jpg', as_gray=True), imread('4_alt.jpg', as_gray=True)])

# Targets for the images
targets = np.array([1, 1, 2, 2, 3, 3, 4, 4])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(images.reshape(len(images), -1), targets, test_size=0.2)

from sklearn.svm import SVC
svc = SVC(kernel='linear')
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

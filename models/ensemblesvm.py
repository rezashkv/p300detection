import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
import mat73
from utils.metrics import accuracy_score

X_train = np.loadtxt('../Data/X_train_A_pca.npy.gz')
y_train = np.loadtxt('../Data/y_train.npy.gz')
X_test = np.loadtxt('../Data/X_test_A_pca.npy.gz')
target = mat73.loadmat('../Data/Test_A.mat')['test_str_A']

clf = BaggingClassifier(SVC(kernel='rbf', class_weight={1: 5}), n_estimators=10)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(accuracy_score(y_pred, target))

from scipy.io import loadmat
import mat73
from sklearn.decomposition import PCA

train_data = loadmat('Train_A.mat')
test_data = mat73.loadmat('Test_A.mat')
X_test = test_data['Epoch_test_A']
X_train = train_data['Epoch_train_A']
del train_data

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1] * X_train.shape[2]))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1] * X_test.shape[2]))

pca = PCA(n_components=200)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

np.savetxt('X_train_A_pca.npy.gz', X_train)
np.savetxt('X_test_A_pca.npy.gz', X_test)

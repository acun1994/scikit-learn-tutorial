# Dependencies
import numpy as np
from sklearn.cluster import KMeans

# Dataset import
from sklearn import datasets

# Dataset preprocessors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

# Data visualisation - Dependencies
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

# Color mapping
LABEL_COLOR_MAP = {0 : 'r',
                   1 : 'k',
                   2 : 'b'}

# Plot drawing function
def visualise(X,y, title, number):
    label_color = [LABEL_COLOR_MAP[l] for l in y]

    fig = plt.figure(num = number, figsize=(8, 6))
    ax = Axes3D(fig, elev=-150, azim=110)
    X_reduced = PCA(n_components=3).fit_transform(X)
    ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=label_color,
            cmap=plt.cm.Set1, edgecolor='k', s=40)
    ax.set_title("First three PCA directions")
    ax.set_xlabel("1st eigenvector")
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("2nd eigenvector")
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("3rd eigenvector")
    ax.w_zaxis.set_ticklabels([])
    plt.title(title)

# Dataset loading                   
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Dataset scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Dataset splitting
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.33)

# Training
kmeans = KMeans(n_clusters=3).fit(X_train)

# Prediction
predict_y = []
for i in range(len(X)):
    predict_me = np.array(X_scaled[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    predict_y.append(prediction[0])

# Visualization
visualise(X,y, "Real", 1)
visualise(X,predict_y, "Predict", 2)

plt.show()
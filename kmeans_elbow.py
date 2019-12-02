import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist

from sklearn.preprocessing import MinMaxScaler


def elbow(site):
    df = pd.read_csv('./data/'+site+'_data.csv')

    # sns.lmplot("counts", "speed", data=df, fit_reg=False, height=6)
    # plt.show()
    # plt.title('kmean plot')
    # plt.xlabel('x')
    # plt.ylabel('y')

    data_points = df.iloc[:, 1:].values
    # kmeans = KMeans(n_clusters=4).fit(data_points)
    # df['cluster_id'] = kmeans.labels_
    # sns.lmplot('counts', 'speed', data=df, fit_reg=False,  # x-axis, y-axis, data, no line
    #            scatter_kws={"s": 150}, # marker size
    #            hue="cluster_id") # color
    # plt.title('after kmean clustering')
    # plt.show()

    # k means determine k
    distortions = []
    for k in range(1,10):
        kmeanModel = KMeans(n_clusters=k).fit(data_points)
        kmeanModel.fit(data_points)
        distortions.append(sum(np.min(cdist(data_points, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / data_points.shape[0])

    # Plot the elbow
    plt.plot(range(1,10), distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k : '+ site)
    plt.show()


sitelist = ["네이트판", "도탁스", "아프리카TV", "오유", "웃긴대학", "이종락싸", "인벤",
            "인스티즈", "일베", "쭉빵여시", "트위터"]

for site in sitelist:
    elbow(site)
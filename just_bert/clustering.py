from bert_serving.client import BertClient
from sklearn.cluster import KMeans
from sklearn import manifold
import matplotlib.pyplot as plt
import numpy as np

bc = BertClient()
n_clusters = 20

# Read file and make embedding
f = open('../final.txt', 'r')
words = []
while True:
    line = f.readline()
    if not line:
        break
    words.append(line[:-1])  # remove the last '\n'
embs = bc.encode(words)  # embeddings

words = np.array(words)
embs = np.array(embs)


# Clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embs)  # clustering
labels = kmeans.labels_


# Write file
for i in range(n_clusters):
    pos = np.where(labels==i)
    w = words[pos[0]]
    f_name = 'out_%s.txt' % i
    f_out = open(f_name, 'w')
    for j in w:
        f_out.write(j+'\n')
    f_out.close()

f.close()

# Visualization
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0).fit_transform(embs)
figure, axesSubplot = plt.subplots()
axesSubplot.scatter(tsne[:, 0], tsne[:, 1], c=labels)



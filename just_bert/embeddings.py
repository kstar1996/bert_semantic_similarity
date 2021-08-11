from bert_serving.client import BertClient
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def get_cosine_similarity(feature_vec_1, feature_vec_2):
    return cosine_similarity(feature_vec_1.reshape(1, -1), feature_vec_2.reshape(1, -1))[0][0]


bc = BertClient()
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
np_embs = np.array(embs)

# Write file
f_out = open('embeddings.txt', 'w')
for j in embs:
    f_out.write(str(j)+'\n')
f_out.close()

f.close()



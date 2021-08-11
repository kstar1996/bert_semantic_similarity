from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pickle


def get_cosine_similarity(feature_vec_1, feature_vec_2):
    return cosine_similarity(feature_vec_1.reshape(1, -1), feature_vec_2.reshape(1, -1))[0][0]


model = SentenceTransformer('paraphrase-TinyBERT-L6-v2')

f = open('../final.txt', 'r')
word_emb = []
while True:
    line = f.readline()
    if not line:
        break
    word_emb.append([line[:-1], model.encode(line[:-1])])

f_out = open('embeddings.txt', 'w')
for j in word_emb:
    f_out.write(str(j)+'\n')
f_out.close()

# pickle file
with open('embeddings.pkl', 'wb') as f_pkl:
    pickle.dump(word_emb, f_pkl)

f.close()


from bert_serving.client import BertClient
from sklearn.metrics.pairwise import cosine_similarity


def get_cosine_similarity(feature_vec_1, feature_vec_2):
    return cosine_similarity(feature_vec_1.reshape(1, -1), feature_vec_2.reshape(1, -1))[0][0]


bc = BertClient()
# Read file and make embedding

query = str(input("Input search query: "))
query_emb = bc.encode([query])

f = open('../final.txt', 'r')
word_emb = []
while True:
    line = f.readline()
    if not line:
        break
    word_emb.append([line[:-1], get_cosine_similarity(query_emb, bc.encode([line[:-1]]))])

sim_scores = sorted(word_emb, key=lambda x: x[1], reverse=True)
sim_scores = sim_scores[1:21]
top = [i[0] for i in sim_scores]

f_out = open(query+'_cosine_sim.txt', 'w')
for n in top:
    f_out.write(str(n)+'\n')
f_out.close()

f.close()



from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pickle
import logging
from time import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_cosine_similarity(feature_vec_1, feature_vec_2):
    return cosine_similarity(feature_vec_1.reshape(1, -1), feature_vec_2.reshape(1, -1))[0][0]


model = SentenceTransformer('paraphrase-TinyBERT-L6-v2')

query = str(input("Input search query: "))
ts = time()
query_emb = model.encode([query])

with open('embeddings.pkl', 'rb') as emb_f:
    mynewlist = pickle.load(emb_f)

for i in range(len(mynewlist)):
    mynewlist[i][1] = get_cosine_similarity(mynewlist[i][1], query_emb)

sim_scores = sorted(mynewlist, key=lambda x: x[1], reverse=True)
sim_scores = sim_scores[1:21]
top = [i[0] for i in sim_scores]

f_out = open('top20/'+query+'_transformer_sim.txt', 'w')
for n in top:
    f_out.write(str(n)+'\n')
f_out.close()

# for checking time
logging.info('Took %s seconds', time() - ts)



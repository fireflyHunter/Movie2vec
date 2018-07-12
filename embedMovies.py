from gensim.models import Word2Vec
from collections import Counter
import numpy as np
import pickle
m20_model = "w2v_models/m20_w2v.model"
m10_model = "w2v_models/m10_w2v.model"
def load_movie_sents(file):
    data = []
    with open(file,'r') as f:
        for i,line in enumerate(f.readlines()):
            if i%2 == 0:
                data.append(line.strip().split(","))
    return data


def train_w2v(data, modelFile):
    print("Training on {} sentences".format(len(data)))
    model = Word2Vec(data, size=300, window=5, min_count=5, workers=4)
    model.save(modelFile)
    
def get_low_freq_word(data):
    all_words = []
    low_feq_words = []
    for sent in data:
        words = sent
        all_words.extend(words)
    counts = Counter(all_words)
    for k,v in counts.items():
        if v < 5:
            low_feq_words.append(k)
    print(low_feq_words)
    
        
def process_train_w2v():
    m20_sents = "ml-20m/ml-20m/movie_sents_low_count_removed.txt"
    m10_sents = "ml-10m/ml-10M100K/movie_sents_low_count_removed.txt"
    
    m20_data = load_movie_sents(m20_sents)
    m10_data = load_movie_sents(m10_sents)
    train_w2v(m20_data, m20_model)
    train_w2v(m10_data, m10_model)


def dump_w2v(model,output_file):
    map_vecs = []
    i2w = model.wv.index2entity
    i2w = [int(x) for x in i2w]
    
    vecs = model.wv.vectors
    for i,v in zip(i2w,vecs):
        map_vecs.append((i,v))
    sorted_map = sorted(map_vecs,key=lambda t: t[0])
    sorted_vecs = [x[1] for x in sorted_map]
    sorted_vecs = np.asarray(sorted_vecs)
    sorted_id = [x[0] for x in sorted_map]
    sorted_id = np.asarray(sorted_id)
    # data=np.asarray([sorted_vecs,sorted_id])
    sorted_vecs.dump(output_file)
    sorted_id.dump(output_file+"index_map")

    
    
if __name__ == "__main__":
    
    m20 = Word2Vec.load(m20_model)
    m10 = Word2Vec.load(m10_model)

    dump_w2v(m20,"ml-20m/ml-20m-vectors")
    dump_w2v(m10,"ml-10m/ml-10M100K/ml-10m-vectors")

    # print(model["1193"])
    

    
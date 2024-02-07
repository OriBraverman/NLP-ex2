import os
import gensim.downloader as dl
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from dim_reduction_utils import reduce_dim_fit, reduce_dim_infer
import sys
import codecs

def read_data(fname):
    for line in codecs.open(fname):
        line = line.strip().split()
        tagged = [x.rsplit("/",1) for x in line]
        yield tagged
def train_knn(train_data,model):
    data = []
    labels = []
    for item in train_data:
        if len(item) == 2:
            word, tag = item
            if word in model:
                data.append(model[word])
                labels.append(tag)
    pca = reduce_dim_fit(data)
    X = reduce_dim_infer(pca, data)
    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(X, labels)
    return knn, pca

def main():
    google_model = dl.load("word2vec-google-news-300")
    train_path = 'part 2/ner/train'
    tagged_data = read_data(train_path)
    clf, dim_red = train_knn(tagged_data,model=google_model)

if __name__=='__main__':
    main()
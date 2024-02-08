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
def build_k_nearest(train_data, model):
    data = []
    labels = []
    for line in train_data:
        for pair in line:
            if len(pair) != 2:
                continue
            word, tag = pair
            if word in model:
                data.append(model[word])
                labels.append(tag)
    pca = reduce_dim_fit(data)
    X = reduce_dim_infer(pca, data)
    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(X, labels)
    return knn, pca
def tag_test_data(test_data,emb_model, clf, pca,output_file):
    tagged_data = []
    with open(output_file,'w') as f:
        for line in test_data:
            tagged_line = ""
            for word in line:
                if word[0] in emb_model:
                    tag = clf.predict(reduce_dim_infer(pca, [emb_model[word[0]]]))[0]
                else:
                    tag = "O"
                tagged_line+=f'{word[0]}/{tag} '
                tagged_data.append((word[0], tag))
                #tagged_lines.append(tagged_line)
            f.writelines([tagged_line,"\n"])
    return tagged_data
def main():
    google_model = dl.load("word2vec-google-news-300")
    train_path = './ner/train'
    val_path = './ner/val'
    test_path = './ner/test.blind'
    tagged_data = read_data(train_path)
    print("training")
    clf, dim_red = build_k_nearest(tagged_data, model=google_model)
    print("done training")
    val=tag_test_data(read_data(val_path),emb_model=google_model,clf=clf,pca=dim_red,output_file='val_pred')
    print(val)
    test=tag_test_data(read_data(test_path), emb_model=google_model, clf=clf, pca=dim_red, output_file='predicted_file')
    print(test)
if __name__=='__main__':
    main()
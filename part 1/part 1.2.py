import os
import gensim.downloader as dl
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier

#  Constants
model_google = dl.load("word2vec-google-news-300")
TRAIN_DATA_PATH = "data/ass1-tagger-train"
TEST_DATA_PATH = "data/ass1-tagger-dev-input"
OUTPUT_PATH = "output/part1.2-dev-output.txt"
LABELED_TEST_DATA_PATH = "data/ass1-tagger-dev"

def create_output_file(output_path):
    if not os.path.exists("output"):
        os.makedirs("output")
    with open(output_path, "w") as f:
        f.write("")

def read_train_data(file_path):
    with open(file_path, "r") as f:
        data = f.read().split()
    return [tuple(x.split("/")) for x in data]

def train_tagger(train_data):
    """use the method of K-nearest neighbors to tag the words in the train data"""
    X = []
    y = []
    for item in train_data:
        if len(item) == 2:
            word, tag = item
            if word in model_google:
                X.append(model_google[word])
                y.append(tag)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X, y)
    return knn

def tag_test_data(test_data, word_tag):
    """map each word in the test data to tuple (word, tag) based on word_tag dictionary"""
    tagged_data = []
    for word in test_data:
        if word in word_tag:
            tagged_data.append((word, word_tag[word]))
        else:
            # if the word is not in the train data, check the 10 most similar words to it in the model
            if word in model_google:
                most_similar = model_google.most_similar(word, topn=10)
                for similar_word, _ in most_similar:
                    if similar_word in word_tag:
                        tagged_data.append((word, word_tag[similar_word]))
                        break
            else:
                # if no similar word is in the train data, tag the word as NN
                tagged_data.append((word, "NN"))
    return tagged_data

def check_accuracy():
    same = 0
    total = 0
    with open(LABELED_TEST_DATA_PATH, "r") as f1, open(OUTPUT_PATH, "r") as f2:
        for line1, line2 in zip(f1, f2):
            line1 = line1.split()
            line2 = line2.split()
            for tag1, tag2 in zip(line1, line2):
                if tag1 == tag2:
                    same += 1
                total += 1
    print(f"Accuracy: {same/total}")
def main():
    train_data = read_train_data(TRAIN_DATA_PATH)

    word_tag = train_tagger(train_data)

    create_output_file(OUTPUT_PATH)

    with open(TEST_DATA_PATH, "r") as f1, open(OUTPUT_PATH, "w") as f2:
        for line in f1:
            test_data = line.split()
            tagged_test_data = tag_test_data(test_data, word_tag)
            # write the output to file in a format w1/t1 w2/t2 ...
            f2.write(" ".join(["{}/{}".format(word, tag) for word, tag in tagged_test_data]))
            f2.write("\n")

    check_accuracy()

if __name__ == "__main__":
    main()

import numpy as np
import os
import gensim.downloader as dl
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from dim_reduction_utils import reduce_dim_fit, reduce_dim_infer
from transformers import RobertaTokenizer, RobertaModel, pipeline

#  Constants
TRAIN_DATA_PATH = "data/ass1-tagger-train"
TEST_DATA_PATH = "data/ass1-tagger-dev-input"
OUTPUT_PATH = "output/part1.3-dev-output.txt"
LABELED_TEST_DATA_PATH = "data/ass1-tagger-dev"

# Global Variables
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')
unmasker = pipeline('fill-mask', model='roberta-base')

def create_output_file(output_path):
    if not os.path.exists("output"):
        os.makedirs("output")
    with open(output_path, "w") as f:
        f.write("")

def read_train_data(file_path):
    with open(file_path, "r") as f:
        data = f.read().split()
    return [tuple(x.split("/")) for x in data]

def train_data(file_path):
    """use the method of K-nearest neighbors to tag the words in the train data"""
    X = []
    y = []
    with open(file_path, "r") as f:
        # read 1 line at a time
        lines = f.readlines()
        for line in lines:
            if lines.index(line) > 20000:
                break
            print(f"line {lines.index(line)} from {len(lines)}")
            # if there is word without / remove it
            line = " ".join([x for x in line.split() if "/" in x])
            curr_X = [x.split("/")[0] for x in line.split()]
            line_without_tags = " ".join(curr_X)
            curr_y = [x.split("/")[1] for x in line.split()]
            # get the word vectors for each word in the line
            curr_X = get_word_vector_list_from_sentence(line_without_tags)
            X.extend(curr_X)
            y.extend(curr_y)
    # if there is item in x = np.zeros(768) remove it from X and y
    for i in range(len(X)):
        if i >= len(X):
            break
        if np.all(X[i] == 0):
            X.pop(i)
            y.pop(i)

    pca = reduce_dim_fit(X)
    X = reduce_dim_infer(pca, X)
    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(X, y)
    return knn, pca


def get_word_vector_list_from_sentence(sentence):
    max_sequence_length = tokenizer.model_max_length
    word_vectors = []
    # divide the sentence into chunks <= max_sequence_length
    # if there is a word that passes max_sequence_length, move it to the next chunk
    sentence_list = sentence.split()
    last_word = 0
    for i in range(len(sentence_list)):
        chunk = " ".join(sentence_list[last_word:i+1])
        if len(tokenizer(chunk)["input_ids"]) > max_sequence_length:
            word_vectors.extend(get_word_vector_list_from_chunk(" ".join(sentence_list[last_word:i])))
            last_word = i
    word_vectors.extend(get_word_vector_list_from_chunk(" ".join(sentence_list[last_word:])))
    return word_vectors


def get_word_vector_list_from_chunk(sentence):
    enc_input = tokenizer(sentence, return_tensors='pt')
    output = model(enc_input["input_ids"])
    sentence_list = sentence.split()
    word_vectors = []
    curr_token_index = 0
    for i in range(len(sentence_list)):
        word = sentence_list[i] if i == 0 else "Ġ" + sentence_list[i]
        token_id = tokenizer.convert_tokens_to_ids(word)
        if token_id not in enc_input["input_ids"][0].tolist():
            word_vectors.append(np.zeros(768))
            continue
        while tokenizer.convert_tokens_to_ids(word) != enc_input["input_ids"][0].tolist()[curr_token_index] and curr_token_index < len(enc_input["input_ids"][0]):
            curr_token_index += 1
        word_vectors.append(output.last_hidden_state.detach().numpy()[0][curr_token_index])
        curr_token_index += 1
    return word_vectors

def get_word_vector_from_sentence(sentence, word):
    enc_input = tokenizer(sentence, return_tensors='pt')
    output = model(enc_input["input_ids"])
    word_token = tokenizer.convert_tokens_to_ids(word)
    word_index = enc_input["input_ids"][0].tolist().index(word_token)
    word_vector = output.last_hidden_state.detach().numpy()[0][word_index]
    return word_vector


def tag_test_data(sentence, knn_tag, pca):
    """map each word in the test data to tuple (word, tag) based on word_tag dictionary"""
    tagged_data = []
    word_vectors = get_word_vector_list_from_sentence(sentence)
    word_vectors = reduce_dim_infer(pca, word_vectors)
    tags = knn_tag.predict(word_vectors)
    for word, tag in zip(sentence.split(), tags):
        tagged_data.append((word, tag))
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
    knn_tag, pca = train_data(TRAIN_DATA_PATH)

    create_output_file(OUTPUT_PATH)

    count = 0
    with open(TEST_DATA_PATH, "r") as f1, open(OUTPUT_PATH, "w") as f2:
        for line in f1:
            count += 1
            print(f"line {count} from 1500")
            tagged_test_data = tag_test_data(line, knn_tag, pca)
            # write the output to file in a format w1/t1 w2/t2 ...
            f2.write(" ".join(["{}/{}".format(word, tag) for word, tag in tagged_test_data]))
            f2.write("\n")

    check_accuracy()

if __name__ == "__main__":
    main()

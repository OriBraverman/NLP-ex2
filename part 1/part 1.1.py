import os
from collections import Counter

#  Constants
TRAIN_DATA_PATH = "data/ass1-tagger-train"
TEST_DATA_PATH = "data/ass1-tagger-dev-input"
OUTPUT_PATH = "output/part1.1-dev-output.txt"
LABELED_TEST_DATA_PATH = "data/ass1-tagger-dev"

# Global Variables
tags_freq = Counter()

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
    """get for each word the most common tag"""
    word_tag = {}
    for item in train_data:
        if len(item) == 2:
            word, tag = item
            tags_freq[tag] += 1
            word_tag.setdefault(word, Counter())[tag] += 1
    return {word: tags.most_common(1)[0][0] for word, tags in word_tag.items()}

def tag_test_data(test_data, word_tag):
    """map each word in the test data to tuple (word, tag) based on word_tag dictionary"""
    tagged_data = [(word, word_tag.get(word, tags_freq.most_common(1)[0][0])) for word in test_data]
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

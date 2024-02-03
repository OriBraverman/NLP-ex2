#train = [(word,ner)]
eu = 'EU/I-ORG'

def raw_word_to_tuple(raw_word):
    word, labels = raw_word.split('/')
    extra, main = labels.split('-')
    return (word, main)

def create_dataset(raw_text):
    dataset = []
    words = raw_text.split()
    for word in words:
        dataset.append(raw_word_to_tuple(raw_word=word))
    return dataset

def read_file_to_dataset(filepath):
    with open(filepath, mode='r') as f:
        return create_dataset(f.read())

if __name__=='__main__':
    word, tag = raw_word_to_tuple(eu)
    print(word, tag)
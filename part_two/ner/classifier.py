import numpy as np
from transformers import RobertaTokenizer, RobertaModel, pipeline
from part_zero import get_word_vector_from_sentence
from ner_train import raw_word_to_tuple
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')
unmasker = pipeline('fill-mask', model='roberta-base')
def create_neighborhood():
    neighbors = []
    for sentence in sentences_train_data:
        for word in sentence:
            word,tag= raw_word_to_tuple(raw_word)
            neighbors.append((get_word_vector_from_sentence(sent,word),tag))

    return words


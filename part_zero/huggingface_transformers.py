import numpy as np
from transformers import RobertaTokenizer, RobertaModel, pipeline


tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')
unmasker = pipeline('fill-mask', model='roberta-base')


def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def get_word_vector_from_sentence(sentence, word):
    enc_input = tokenizer(sentence, return_tensors='pt')
    output = model(enc_input["input_ids"])
    word_token = tokenizer.convert_tokens_to_ids(word)
    word_index = enc_input["input_ids"][0].tolist().index(word_token)
    word_vector = output.last_hidden_state.detach().numpy()[0][word_index]
    return word_vector

# Extracting vector for "am" and "<mask>"
text = "I am so <mask>"
am_vector = get_word_vector_from_sentence(text, "Ġam")
mask_vector = get_word_vector_from_sentence(text, "<mask>")  # other way: mask_token = tokenizer.mask_token_id


# Extracting top-5 word predictions for "am" and "<mask>" and their probabilities
print("Top-5 word predictions for 'am':")
for pred in unmasker("I <mask> so", top_k=5):
    print(f"{pred['token_str']}: {pred['score']}")
print("\n")

print("Top-5 word predictions for '<mask>':")
for pred in unmasker("I am so <mask>", top_k=5):
    print(f"{pred['token_str']}: {pred['score']}")
print("\n")

# 2) Find two sentences that share the same word,
# such that the cosine similarity between the word vectors in the two sentences is very high.
sentence1 = "I ate an apple"
sentence2 = "I ate a pear"
word = "Ġate"

word_vector1 = get_word_vector_from_sentence(sentence1, word)
word_vector2 = get_word_vector_from_sentence(sentence2, word)

print(f"Cosine similarity using the word ate between '{sentence1}' and '{sentence2}': "
      f"{cosine_similarity(word_vector1, word_vector2)}\n")

# 3) Find two sentences that share the same word,
# such that the cosine similarity between the word vectors in the two sentences is very low.
sentence1 = "I work at apple every day, even on weekends and holidays"
sentence2 = "This apple is red"
word = "Ġapple"

word_vector1 = get_word_vector_from_sentence(sentence1, word)
word_vector2 = get_word_vector_from_sentence(sentence2, word)

print(f"Cosine similarity using the word apple between '{sentence1}' and '{sentence2}': "
        f"{cosine_similarity(word_vector1, word_vector2)}\n")

# 4) Find a sentence with n words, that is tokenized into m > n tokens by the tokenizer.
# this is the longest english word: pneumonoultramicroscopicsilicovolcanoconiosis
sentence = "The longest english word is pneumonoultramicroscopicsilicovolcanoconiosis"
print(f"The sentence: '{sentence}'")
print(f"Tokenization of '{sentence}': {tokenizer(sentence)['input_ids']}")
print(f"Number of words in sentence: {len(sentence.split())}")
print(f"Number of tokens: {len(tokenizer(sentence)['input_ids'])}")
print("\n")

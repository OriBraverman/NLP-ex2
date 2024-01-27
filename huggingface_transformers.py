from transformers import RobertaTokenizer, RobertaModel, pipeline


tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')

unmasker = pipeline('fill-mask', model=model, tokenizer=tokenizer)

text = "I am so <mask>"

enc_input = tokenizer(text, return_tensors='pt')

# Extract the vector for "am" and "<mask>"
am_index = enc_input["input_ids"][0].tolist().index(2)
mask_index = enc_input["input_ids"][0].tolist().index(tokenizer.mask_token_id)
am_vector = model.base_model.embeddings.word_embeddings(enc_input["input_ids"]).detach().numpy()[0][am_index]
mask_vector = model.base_model.embeddings.word_embeddings(enc_input["input_ids"]).detach().numpy()[0][mask_index]

# Extract the top-5 word predictions for "am" and "<mask>" and their probabilities
am_predictions = unmasker(f"I {tokenizer.mask_token} so")
mask_predictions = unmasker(f"I am so {tokenizer.mask_token}")

print("Top-5 word predictions for 'am':")
for pred in am_predictions:
    print(f"{pred['token_str']}: {pred['score']}")
print("Top-5 word predictions for '[MASK]':")
for pred in mask_predictions:
    print(f"{pred['token_str']}: {pred['score']}")

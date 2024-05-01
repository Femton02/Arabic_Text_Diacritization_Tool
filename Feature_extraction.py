from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import numpy as np


def embidding_bert(sentences, max_seq_length=512):
    model_name = "asafaya/bert-base-arabic"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    tokens1 = tokenizer(sentences, return_tensors="pt", padding=True)
    outputs1 = model(**tokens1)
    word_embeddings1 = outputs1.last_hidden_state

    newList = []
    for i in range(word_embeddings1.shape[1]):
        # Vector Embedding of one word
        word_embedding_vector = word_embeddings1[:, i].tolist()
        newList.append(word_embedding_vector[0])

    return newList


# def TF_IDF(sentences):
#     vectorizer = TfidfVectorizer()
#     X = vectorizer.fit_transform(sentences)
#     return X


def one_hot_encode_letter(letter, ARABIC_LETTERS_LIST):
    letter_to_index = {letter: i for i, letter in enumerate(ARABIC_LETTERS_LIST)}

    one_hot_vector = np.zeros(len(ARABIC_LETTERS_LIST))

    if letter in letter_to_index:
        one_hot_vector[letter_to_index[letter]] = 1

    return one_hot_vector

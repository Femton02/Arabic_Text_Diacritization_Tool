import nltk
from nltk.tokenize import sent_tokenize
import re
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from transformers import AutoTokenizer, AutoModel
import torch


def vocab(ALL_CHARACTERS):
    CHAR_TO_INDEX = {char: index for index, char in enumerate(ALL_CHARACTERS)}
    INDEX_TO_CHAR = {index: char for index, char in enumerate(ALL_CHARACTERS)}
    return CHAR_TO_INDEX, INDEX_TO_CHAR


# cleaning the text from non arabic letters
def clean_Text(text):
    text = re.sub(r"[0-9٠-٩()/:/-{}\[\]»«]", "", text)  # remove non arabic letters
    text = re.sub(r"\s[أ-ي]\s", "", text)  # remove single char
    text = re.sub(r"[\s]+", " ", text)  # remove extra spaces
    return text


# removing diacritics from the text
def remove_diactrics(text):
    text = re.sub(r"[\u064B-\u065F]", "", text)
    return text


# tokenizing the text
def tokenize(text):
    sentences = []
    tokenized = tokenize_arabic_sentences(text)
    for sentence in tokenized:
        word_list = nltk.word_tokenize(sentence)
        sentences.append(word_list)
    return sentences

def tokenize_arabic_sentences(text):
    # Define the punctuation marks for sentence splitting
    punctuation_marks = r'[.؛؟!?،]'
    # Split the text into smaller sentences using the specified punctuation marks
    sentences = re.split(punctuation_marks, text)
    # Remove empty strings from the resulting list
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    return sentences
    
def remove_empty_sentences(sentences):
    for sentence in sentences:
        if len(sentence) == 0:
            sentences.remove(sentence)
        for word in sentence:
            if len(word) == 0:
                sentence.remove(word)
    return sentences


def Get_Labels(sentences, diacritic2id):
    labels = []
    for sentence in sentences:
        wordslabel = []
        for word in sentence:
            charlabel = []
            for char in word:
                if char in diacritic2id:
                    charlabel.append(diacritic2id[char])
            wordslabel.append(charlabel)
        labels.append(wordslabel)
    return labels
        

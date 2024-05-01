import Preprocessing
import sys
import os
import InOut
import pickle as pkl
import Feature_extraction
import Model
import InOut

arabic_letters = InOut.read_file_frompickle("constant", "/arabic_letters.pickle")
diacritics = InOut.read_file_frompickle("constant", "/diacritics.pickle")
diacritics2idx = InOut.read_file_frompickle("constant", "/diacritic2id.pickle")
Characters2idx = {char: idx for idx, char in enumerate(arabic_letters)}

file_path_Train = "dataset/train.txt"
file_content_Train = InOut.read_file_content(file_path_Train)
file_content_aftercleaning_Train = Preprocessing.clean_Text(file_content_Train)
wholesentences_Train = Preprocessing.tokenize(file_content_aftercleaning_Train)
sentences_Train = wholesentences_Train[:50]
labels = Preprocessing.Get_Labels(sentences_Train, diacritics2idx)
max_seq_length = max([len(sentence) for sentence in sentences_Train])
bert_embedding = InOut.read_file_frompickle("dataset/", "Train_bert_embedding.pickle")

wordmodel = Model.LSTM_BID(max_seq_length, bert_embedding, len(diacritics2idx))
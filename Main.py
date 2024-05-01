import Preprocessing
import sys
import os
import InOut
import pickle as pkl
import Feature_extraction
import Model
import numpy as np

arabic_letters = InOut.read_file_frompickle("constant", "/arabic_letters.pickle")
diacritics = InOut.read_file_frompickle("constant", "/diacritics.pickle")
diacritics2idx = InOut.read_file_frompickle("constant", "/diacritic2id.pickle")
Characters2idx = {char: idx for idx, char in enumerate(arabic_letters)}
print(arabic_letters)
print(diacritics)
print(diacritics2idx)
print(Characters2idx)
######################################################################################################################Train dataset
# file_path_Train = "dataset/train.txt"
# file_content_Train = InOut.read_file_content(file_path_Train)
# file_content_aftercleaning_Train = Preprocessing.clean_Text(file_content_Train)
# InOut.create_file(
#     "dataset/file_content_aftercleaning_Train.txt", file_content_aftercleaning_Train
# )
# # file_content_afterremovingdia_Train=Preprocessing.remove_diactrics(file_content_aftercleaning_Train)
# # InOut.create_file('dataset/file_content_afterremovingdia_Train.txt', file_content_afterremovingdia_Train)
# sentences_Train = Preprocessing.tokenize(file_content_aftercleaning_Train)

# sentences_Train = [sentence.strip() for sentence in sentences_Train if sentence.strip()]
# labels = Preprocessing.Get_Labels(sentences_Train, diacritics2idx)
# InOut.create_fileformatted("dataset/labels_Train.txt", labels)
# print("Train_bert_embedding Started")
# InOut.create_fileformatted("dataset/sentences_Train.txt", sentences_Train)
# Train_bert_embedding = Feature_extraction.embidding_bert(sentences_Train)
# InOut.write_Pickle("dataset/", "Train_bert_embedding.pickle", Train_bert_embedding)

# Train_TFIDF_embedding = Feature_extraction.TF_IDF(sentences_Train)
# InOut.create_fileformatted('dataset/Train_TFIDF_embedding.pickle', Train_TFIDF_embedding)
###############################################################################################################val dataset
file_path_Val = "dataset/val.txt"
file_content_Val = InOut.read_file_content(file_path_Val)
file_content_aftercleaning_Val = Preprocessing.clean_Text(file_content_Val)
InOut.create_file(
    "dataset/file_content_aftercleaning_Val.txt", file_content_aftercleaning_Val
)
# file_content_afterremovingdia_val=Preprocessing.remove_diactrics(file_content_aftercleaning_Val)
# InOut.create_file('dataset/file_content_afterremovingdia_val.txt', file_content_afterremovingdia_val)
sentences_Val = Preprocessing.tokenize(file_content_aftercleaning_Val)
sentences_Val = [sentence.strip() for sentence in sentences_Val if sentence.strip()]
InOut.create_fileformatted("dataset/sentences_Val.txt", sentences_Val)

Val_bert_embedding_List = []
for i in range(len(sentences_Val)):
    Val_bert_embedding = Feature_extraction.embidding_bert(sentences_Val[i])
    Val_bert_embedding_List.append(Val_bert_embedding)
InOut.write_Pickle("dataset/", "Val_bert_embedding.pickle", Val_bert_embedding_List)

# Val_TFIDF_embedding = Feature_extraction.TF_IDF(sentences_Val)
InOut.create_file("dataset/Val_TFIDF_embedding.pickle", Val_TFIDF_embedding)

###############################################################################################################

print(Feature_extraction.one_hot_encode_letter("ا", arabic_letters))

print(Feature_extraction.one_hot_encode_letter("ي", arabic_letters))

import Preprocessing
import sys
import os
import InOut
import pickle as pkl
import Feature_extraction
import Model


arabic_letters = InOut.read_file_frompickle("constant", "/arabic_letters.pickle")
diacritics = InOut.read_file_frompickle("constant", "/diacritics.pickle")
diacritics2idx = InOut.read_file_frompickle("constant", "/diacritic2id.pickle")
Characters2idx = {char: idx for idx, char in enumerate(arabic_letters)}
print(arabic_letters)
print(diacritics)
print(diacritics2idx)
print(Characters2idx)
# read from pickle

Train_bert_embedding = InOut.read_file_frompickle(
    "dataset", "/Train_bert_embedding.pickle"
)


Val_bert_embedding = InOut.read_file_frompickle("dataset", "/Val_bert_embedding.pickle")


file_path_Train = "dataset/train.txt"
file_content_Train = InOut.read_file_content(file_path_Train)
file_content_aftercleaning_Train = Preprocessing.clean_Text(file_content_Train)
InOut.create_file(
    "dataset/file_content_aftercleaning_Train.txt", file_content_aftercleaning_Train
)
# file_content_afterremovingdia_Train=Preprocessing.remove_diactrics(file_content_aftercleaning_Train)
# InOut.create_file('dataset/file_content_afterremovingdia_Train.txt', file_content_afterremovingdia_Train)


sentences_Train = Preprocessing.tokenize(file_content_aftercleaning_Train)

sentences_Train = [sentence.strip() for sentence in sentences_Train if sentence.strip()]
labels_train = Preprocessing.Get_Labels(sentences_Train, diacritics2idx)


file_path_Val = "dataset/val.txt"
file_content_Val = InOut.read_file_content(file_path_Val)
file_content_aftercleaning_Val = Preprocessing.clean_Text(file_content_Val)
InOut.create_file(
    "dataset/file_content_aftercleaning_Val.txt", file_content_aftercleaning_Val
)


sentences_val = Preprocessing.tokenize(file_content_aftercleaning_Val)

sentences_val = [sentence.strip() for sentence in sentences_val if sentence.strip()]
labels_val = Preprocessing.Get_Labels(sentences_val, diacritics2idx)


# ###############################################################

# model = Model.LSTM_BID(512, 768, 15)
# model.summary()


file_path_Train = "dataset/train.txt"
file_content_Train = InOut.read_file_content(file_path_Train)
file_content_aftercleaning_Train = Preprocessing.clean_Text(file_content_Train)
InOut.create_file(
    "dataset/file_content_aftercleaning_Train.txt", file_content_aftercleaning_Train
)
# file_content_afterremovingdia_Train=Preprocessing.remove_diactrics(file_content_aftercleaning_Train)
# InOut.create_file('dataset/file_content_afterremovingdia_Train.txt', file_content_afterremovingdia_Train)
sentences_Train = Preprocessing.tokenize(file_content_aftercleaning_Train)
max_sequence_length = max(len(sequence) for sequence in sentences_Train)

print(max_sequence_length)


max_sequence_length = 5957


model = Model.LSTM_BID(max_sequence_length, Train_bert_embedding, len(diacritics))

model.summary()

Model.train_Lstm(
    model,
    Train_bert_embedding,
    labels_train,
    Val_bert_embedding,
    labels_val,
    epochs=1,
    batch_size=512,
)

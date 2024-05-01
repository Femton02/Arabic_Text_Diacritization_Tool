import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
import numpy as np
import pickle
import re
import os

# Load the preprocessed dataset
rowdata = pd.read_csv("dataset/sentences.txt", sep="\t")

# Extract sentences using regex
sentence_pattern = re.compile(r'<s>(.*?)<\/s>')
matches = rowdata.apply(lambda row: re.findall(sentence_pattern, row['sentence']), axis=1)
sentences = [match[0].strip() for match in matches]

# Define functions for different feature extraction methods
def extract_bow_features(text):
    vectorizer = CountVectorizer()
    features = vectorizer.fit_transform(text)
    return features, vectorizer

def extract_tfidf_features(text):
    vectorizer = TfidfVectorizer()
    features = vectorizer.fit_transform(text)
    return features, vectorizer

def extract_word_embeddings(text, embedding_dim=100, window=5, min_count=1):
    # Assuming 'text' is a list of tokenized sentences
    model = Word2Vec(sentences=text, vector_size=embedding_dim, window=window, min_count=min_count)
    
    # Create sentence embeddings by averaging word embeddings
    embeddings = []
    for sentence in text:
        sentence_embedding = np.mean([model.wv[word] for word in sentence if word in model.wv], axis=0)
        embeddings.append(sentence_embedding)
    
    return np.array(embeddings), model

# Create a folder to store feature extraction results
output_folder = "FeatureExtraction"
os.makedirs(output_folder, exist_ok=True)

# Perform feature extraction for all methods
# 1. Bag of Words (BoW)
bow_features, bow_vectorizer = extract_bow_features(sentences)
bow_save_path = os.path.join(output_folder, "bow_features.npy")
bow_vectorizer_save_path = os.path.join(output_folder, "bow_vectorizer.pkl")
np.save(bow_save_path, bow_features)
with open(bow_vectorizer_save_path, "wb") as bow_vectorizer_file:
    pickle.dump(bow_vectorizer, bow_vectorizer_file)

# 2. TF-IDF
tfidf_features, tfidf_vectorizer = extract_tfidf_features(sentences)
tfidf_save_path = os.path.join(output_folder, "tfidf_features.npy")
tfidf_vectorizer_save_path = os.path.join(output_folder, "tfidf_vectorizer.pkl")
np.save(tfidf_save_path, tfidf_features)
with open(tfidf_vectorizer_save_path, "wb") as tfidf_vectorizer_file:
    pickle.dump(tfidf_vectorizer, tfidf_vectorizer_file)

# 3. Word Embeddings
word_embeddings, word2vec_model = extract_word_embeddings(sentences)
word_embeddings_save_path = os.path.join(output_folder, "word_embeddings.npy")
word2vec_model_save_path = os.path.join(output_folder, "word2vec_model.model")
np.save(word_embeddings_save_path, word_embeddings)
word2vec_model.save(word2vec_model_save_path)

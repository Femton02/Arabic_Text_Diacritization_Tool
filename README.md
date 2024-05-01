# CMP Elective 4 Project: Arabic Diacritization

This repository contains the documentation and code for the Natural Language Processing project undertaken during the Fall 2023 semester at Cairo University's Computer Engineering Department.

## Project Description

Arabic, being one of the most spoken languages globally, presents unique challenges in natural language processing (NLP) due to its rich morphology and the presence of diacritics. The absence of diacritics in Arabic text can lead to ambiguity in both pronunciation and meaning. This project focuses on restoring diacritics to Arabic text to enhance various NLP tasks such as Text-To-Speech systems and machine translation.

## Dataset Description

The dataset consists of discretized Arabic sentences, divided into training, development (dev), and test sets. The training and dev sets are annotated with diacritics, while the test set contains text without diacritization. The dataset sizes are as follows:

- Training set: 50k lines
- Dev set: 2.5k lines
- Test set: 2.5k lines

## Project Pipeline

The project pipeline involves several stages:

1. **Data Preprocessing**: Includes cleaning the data (e.g., removing HTML tags, English letters) and tokenization.
2. **Feature Extraction**: Experimentation with different features, such as Bag of Words, TF-IDF, Word Embeddings, and contextual embeddings, at both word and character levels.
3. **Model Building**: Building machine learning models (e.g., RNN, LSTM, CRF, HMM) to predict diacritics. Multiple models may be employed, considering word-level and character-level encoding.
4. **Model Testing**: Evaluating the best-performing model on the test set.

## Approach

We employed two models:

- **Contextual Character-Diacritic Relation Model**: Captures the relationship between characters and diacritics.
- **Word-Meaning Model**: Models the relationship between words and the meaning of the sentence.

# Fake news detection on social platforms

This is a project demonstrating the accuracy of selected machine learning classification algorithms (NN, SVM, LR, RF, NB, AdaBoost, KNN) combined with word embeddings (BOW, W2V and a custom set of features) in the task of determining whether a comment/post is fake news or not. The only information that is taken into account is the text of the comment - this allows the system to be used on any platform as it does not rely on specific metadata available on individual social networks. Accuracy scores are the average of 10-fold cross-validation.

## Custom set of features

The table below presents a set of linguistic features based on the work of: Pedro Henrique Arruda Faustini and Thiago Ferreira Covões. Fake news detection in multiple platforms and languages. Expert Syst. Appl.,158:113503, 2020.

This set has been extended with several readability measures available in the Textstat library, parts of speech such as modal verbs, prepositions and pronouns, and additionally we took into account non-informative words, punctuation and statistical values for word and sentence length. The values of most of these features represent proportions based on either text or word length and fall within the 0-1 boundary already at the matrix filling stage, but for some features such as the Flesch Readability Index, which operates on a range of values from 0-100, we had to additionally scale some columns to the 0-1 range. This is important because deep learning works best within the 0-1 boundary. The readability measures that were used, such as the Flesch Readability Index, are designed to measure the degree of difficulty in understanding a given text in English. The lower the index, the more difficult the text is to understand. The value is calculated from the number of words, syllables and sentences in the text.

| Id | Feature                 | Id | Feature                 |
|----|-------------------------|----|-------------------------|
| 1  | Flesch reading ease     | 13 | Number of words         |
| 2  | SMOG index              | 14 | Number of sentences     |
| 3  | Coleman-Liau index      | 15 | Adjectives              |
| 4  | Uppercase letters       | 16 | Nouns                   |
| 5  | Exclamation marks       | 17 | Verbs                   |
| 6  | Question marks          | 18 | Adverbs                 |
| 7  | Unique words            | 19 | Prepositions            |
| 8  | Uninformative words     | 20 | Pronouns                |
| 9  | Punctuation             | 21 | Personal pronouns       |
| 10 | Long words              | 22 | Modal verbs             |
| 11 | Average word length     | 23 | Sentiment analysis      |
| 12 | Average sentence length | 24 | Word2Vec representation |

## Results

The Support Vector Machine proved to be the best classifier in this competition, achieving more than 86% accuracy in one of the sets. When it comes to disinformation, it is important that as little as possible fake news is detected as non-threatening messages, as the damage it causes can be considerable, especially in areas such as politics or medicine. The Random Forest classifier certainly deserves a special mention in this respect, as it excelled in disinformation detection by achieving the lowest error on the confusion matrix by detecting the highest percentage of false information (0.69 in the example figure), at the cost of a slightly lower than average accuracy in evaluating safe news (0.83).

| Dataset | Word Embeddings        | NN                   | **SVM**                  | LR                   | RF                   | NB                   | AdaBoost             | KNN 3                | KNN 5                | KNN 7                |
| ------- | ---------------------- | -------------------- | ------------------------ | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- |
| KAGGLE  | BOW                    | 75.6\% <br /> (±1.2) | **79.1\%** <br /> (±1.3) | 77.9\%<br />(±1.2)   | 77.0\% <br /> (±1.5) | 75.7\% <br /> (±1.1) | 74.9\% <br /> (±1.0) | 73.2\% <br /> (±1.5) | 73.0\% <br /> (±2.1) | 72.9\% <br /> (±1.8) |
|         | Word2Vec               | 74.0\% <br /> (±1.6) | 78.0\% <br /> (±1.0)     | 76.4\% <br /> (±1.0) | 77.7\% <br /> (±1.3) | 74.4\% <br /> (±1.3) | 75.0\% <br /> (±0.8) | 74.3\% <br /> (±1.6) | 76.1\% <br /> (±1.7) | 76.7\% <br /> (±1.3) |
|         | Custom set of features | 69.6\% <br /> (±2.6) | 71.4\% <br /> (±1.9)     | 70.7\% <br /> (±1.9) | 73.5\% <br /> (±1.5) | 64.4\% <br /> (±1.5) | 70.3\% <br /> (±1.6) | 67.7\% <br /> (±2.4) | 68.4\% <br /> (±2.5) | 69.3\% <br /> (±1.9) |
| PHEME1  | BOW                    | 84.2\% <br /> (±1.6) | **85.7\%** <br /> (±1.1) | 83.8\% <br /> (±1.5) | 84.4\% <br /> (±1.4) | 78.8\% <br /> (±2.0) | 82.1\% <br /> (±2.1) | 81.1\% <br /> (±1.5) | 79.7\% <br /> (±1.2) | 79.6\% <br /> (±2.1) |
|         | Word2Vec               | 77.8\% <br /> (±1.6) | 77.8\% <br /> (±1.0)     | 74.1\% <br /> (±1.6) | 77.0\% <br /> (±1.7) | 69.6\% <br /> (±1.3) | 73.3\% <br /> (±1.8) | 77.8\% <br /> (±1.5) | 77.6\% <br /> (±1.5) | 78.7\% <br /> (±0.8) |
|         | Custom set of features | 69.6\% <br /> (±1.7) | 69.0\% <br /> (±2.4)     | 68.7\% <br /> (±2.3) | 71.9\% <br /> (±2.1) | 64.9\% <br /> (±2.1) | 68.7\% <br /> (±1.9) | 65.3\% <br /> (±1.6) | 65.7\% <br /> (±1.4) | 66.4\% <br /> (±1.8) |
| PHEME2  | BOW                    | 84.3\% <br /> (±1.8) | **86.1\%** <br /> (±1.3) | 84.2\% <br /> (±1.2) | 85.0\% <br /> (±1.8) | 74.7\% <br /> (±2.3) | 81.6\% <br /> (±1.9) | 81.5\% <br /> (±2.1) | 79.8\% <br /> (±2.3) | 79.7\% <br /> (±3.2) |
|         | Word2Vec               | 79.5\% <br /> (±1.1) | 79.8\% <br /> (±1.2)     | 76.5\% <br /> (±1.1) | 79.1\% <br /> (±1.1) | 73.4\% <br /> (±1.7) | 75.8\% <br /> (±2.4) | 79.7\% <br /> (±1.5) | 80.4\% <br /> (±1.2) | 80.3\% <br /> (±1.0) |
|         | Custom set of features | 72.6\% <br /> (±1.7) | 71.6\% <br /> (±1.3)     | 71.2\% <br /> (±1.3) | 73.9\% <br /> (±1.9) | 60.7\% <br /> (±2.2) | 70.5\% <br /> (±2.4) | 67.6\% <br /> (±1.2) | 68.5\% <br /> (±1.4) | 68.9\% <br /> (±1.6) |

## Software required

- Python
- Anaconda (optional)

## Libraries

- pandas~=1.2.1
- numpy~=1.18.5
- tensorflow~=2.3.0
- nltk~=3.5
- scikit-learn~=0.24.1
- keras~=2.4.0
- gensim~=3.8.3
- seaborn~=0.11.1
- matplotlib~=3.4.2
- textstat~=0.7.1
- shap~=0.39.0

## Datasets

- https://www.kaggle.com/hamditarek/fake-news-detection-on-twitter-eda/data
- https://www.zubiaga.org/datasets/ ("PHEME dataset for Rumour Detection and Veracity Classification" and "PHEME rumour dataset")

## Run instructions

- Install required libraries with a command `pip install -r requirements.txt`
- Download datasets with a `.csv` format containing two columns: `text` (representing a comment/post) and `target` (value 0 or 1, where 0 means that a comment is not a misinformation, and 1 means that it is)
- For datasets available on https://www.zubiaga.org/datasets/ it is necessary to convert `.json` files to a '.csv' format, which can be done with a script `convert.py`. Example of running this script: `python convert.py path/to/folder/containing/json/files result.csv`
- Running programs `bow.py` (uses a Bag of Words), `customised.py` (uses a custom set of linguistic features) or `w2v.py` (uses a Word2Vec representation). For example, running the tests for a BOW representation: `python bow.py data.csv 0_or_1`, where the value at the end means generating martices of confusion and a linguistic features evaluation (the latter only for a custom set of features defined in `customised.py`)

## Sample of generated figures (in Polish)

![bag of words](./images/conf_matrix_first_bow.png)
Confusion matrices (Bag of Words embeddings, first dataset)

![features](./images/features_first.png)
The most important features (Custom features, first dataset)

import re
import string
import sys
from pathlib import Path
from unicodedata import category

import gensim
import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sn
import shap
import textstat
from gensim import downloader
from nltk import PorterStemmer
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers.core import Dense, Dropout
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier

nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")
nltk.download("vader_lexicon")
nltk.download("stopwords")
nltk.download("punkt")

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))
sentiment_analyzer = SentimentIntensityAnalyzer()

pd.options.mode.chained_assignment = None

if len(sys.argv) < 3:
    print(
        "Uruchomienie programu: python program.py dane.csv czy_generowac_wykresy(0 lub 1)"
    )
    exit(1)

if int(sys.argv[2]) == 1:
    generate_figures = True
else:
    generate_figures = False

Path("figures").mkdir(parents=True, exist_ok=True)
dataset_path = sys.argv[1]
dataset_name = Path(dataset_path).stem
df = pd.read_csv(dataset_path, encoding="utf-8")
df = df[["text", "target"]]
text = df["text"]
print("Wczytano " + str(len(df)) + " rekordów.")
print("Liczba nie-plotek: " + str(len(df[(df["target"] == 0)])))
print("Liczba plotek: " + str(len(df[(df["target"] == 1)])))


def initial_cleansing(sentence):
    parsed = re.sub(
        r"http\S+", "", sentence
    )  # to remove links that start with HTTP/HTTPS in the tweet
    parsed = re.sub(r"@\w+|#", "", parsed)  # Remove user @ references and ‘#’ from text
    return parsed


for i, row in df.iterrows():
    text_val = initial_cleansing(row["text"])
    df.at[i, "text"] = text_val

chrs = (chr(i) for i in range(sys.maxunicode + 1))
punctuation = set(c for c in chrs if category(c).startswith("P"))


def text_cleansing(sentence):
    words = word_tokenize(sentence)
    filtered_sentence = [w for w in words if not w.lower() in stop_words]
    filtered_sentence = [lemmatizer.lemmatize(sent) for sent in filtered_sentence]
    # filtered_sentence = [stemmer.stem(sent) for sent in filtered_sentence]
    filtered_sentence = [w for w in filtered_sentence if w not in punctuation]
    return filtered_sentence


embeddings = gensim.downloader.load("glove-twitter-100")


def word2vec_score(cleaned_sentence):
    temp_df = pd.DataFrame()
    for word in cleaned_sentence:
        try:
            word_vec = embeddings[word]
            temp_df = temp_df.append(pd.Series(word_vec), ignore_index=True)
        except:
            pass
    doc_vector = temp_df.mean()  # average of each column(w0, w1, w2,...,wn)
    res = doc_vector.mean()  # average score
    return res


features = pd.DataFrame(
    columns=[
        "readability-flesch",
        "readability-smog",
        "readability-coleman",
        "uppercase",
        "exclamation",
        "question",
        "unique",
        "stopwords",
        "punctuation",
        "long-words",
        "avg-word-length",
        "avg-sentence-length",
        "words",
        "sentences",
        "adjectives",
        "nouns",
        "verbs",
        "adverbs",
        "prepositions",
        "pronouns",
        "personal-pronouns",
        "modals",
        "sentiment",
        "word2vec",
    ]
)
columns_polish = [
    "czytelność-flesch",
    "czytelność-smog",
    "czytelność-coleman",
    "duże litery",
    "wykrzykniki",
    "znaki zapytania",
    "słowa unikalne",
    "słowa nieinformatywne",
    "interpunkcja",
    "długie wyrazy",
    "średnia dł. wyrazu",
    "średnia dł. zdania",
    "liczba słów",
    "liczba zdań",
    "przymiotniki",
    "rzeczowniki",
    "czasowniki",
    "przysłówki",
    "przyimki",
    "zaimki",
    "zaimki personalne",
    "czasowniki modalne",
    "analiza sentymentu",
    "word2vec",
]
for doc in df["text"]:
    doc_length = len(doc)
    word_tokens = word_tokenize(doc)
    word_tokens_length = len(word_tokens)
    cleaned_sentence = text_cleansing(doc)
    cleaned_sentence_length = len(cleaned_sentence)
    temp = list()
    temp.append(textstat.flesch_reading_ease(doc))
    temp.append(textstat.smog_index(doc))
    temp.append(textstat.coleman_liau_index(doc))

    temp.append(sum(1 for c in doc if c.isupper()) / doc_length)
    temp.append(sum(1 for c in doc if c == "!") / doc_length)
    temp.append(sum(1 for c in doc if c == "?") / doc_length)
    temp.append(len(set([word.lower() for word in word_tokens])) / word_tokens_length)

    stopwords_x = [w for w in word_tokens if w in stop_words]
    temp.append(len(stopwords_x) / len(word_tokens))

    count = 0
    for i in doc:
        if i in string.punctuation:
            count += 1
    temp.append(count / doc_length)

    count = 0
    for word in word_tokens:
        if len(word) > 6:
            count += 1
    temp.append(count / word_tokens_length)

    count = sum(len(word) for word in word_tokens)
    temp.append(count / word_tokens_length)

    count = 0
    sentences = sent_tokenize(doc)
    for s in sentences:
        tokens = word_tokenize(s)
        count += len(tokens)
    temp.append(count / len(sentences))

    temp.append(textstat.lexicon_count(doc, removepunct=True))
    temp.append(textstat.sentence_count(doc))

    adjs = 0
    nouns = 0
    verbs = 0
    adverbs = 0
    prepositions = 0
    pronouns = 0
    personal_pronouns = 0
    modals = 0
    tagged = nltk.pos_tag(word_tokens)
    for tag in tagged:
        pos = tag[1]
        if pos.startswith("JJ"):
            adjs += 1
        if pos.startswith("NN"):
            nouns += 1
        if pos.startswith("VB"):
            verbs += 1
        if pos.startswith("RB"):
            adverbs += 1
        if pos.startswith("IN"):
            prepositions += 1
        if pos.startswith("PR"):
            personal_pronouns += 1
        elif pos.startswith("P"):
            pronouns += 1
        if pos.startswith("M"):
            modals += 1

    temp.append(adjs / word_tokens_length)
    temp.append(nouns / word_tokens_length)
    temp.append(verbs / word_tokens_length)
    temp.append(adverbs / word_tokens_length)
    temp.append(prepositions / word_tokens_length)
    temp.append(pronouns / word_tokens_length)
    temp.append(personal_pronouns / word_tokens_length)
    temp.append(modals / word_tokens_length)

    sentiment_score = sentiment_analyzer.polarity_scores(doc)["pos"]
    temp.append(sentiment_score)
    temp.append(word2vec_score(cleaned_sentence))

    features.loc[len(features)] = temp

min_max_scaler = MinMaxScaler()
columns_to_transform = [
    "readability-flesch",
    "readability-smog",
    "readability-coleman",
    "avg-word-length",
    "avg-sentence-length",
    "words",
    "sentences",
    "sentiment",
    "word2vec",
]
features[columns_to_transform] = min_max_scaler.fit_transform(
    features[columns_to_transform]
)

print(features.head())

features["target"] = df["target"]
features = features.dropna()
X_text = features.drop("target", axis=1)
y_values = features["target"]


def create_model():
    model = Sequential()
    model.add(Dense(100, input_dim=(X_text.shape[1]), activation="relu"))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(30, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def generate_heatmaps(values):
    fig, axes = plt.subplots(3, 3)
    fig.tight_layout()
    fig.set_size_inches(8, 6)
    fig.suptitle("Macierze błędów")
    cmap = sn.color_palette("Blues", as_cmap=True)

    axs = axes.flat
    for index, value in enumerate(values):
        df_cm = pd.DataFrame(value[1])
        sn.set(font_scale=0.8)
        g1 = sn.heatmap(
            df_cm,
            vmin=0,
            vmax=1,
            annot=True,
            annot_kws={"size": 14},
            cmap=cmap,
            ax=axs[index],
        )
        g1.set_ylabel("Klasa rzeczywista", fontsize=8)
        g1.set_xlabel("Klasa predykowana", fontsize=8)
        g1.get_xaxis().set_ticklabels(["Nie-plotka", "Plotka"], va="center")
        g1.get_yaxis().set_ticklabels(["Nie-plotka", "Plotka"], va="center")
        g1.tick_params(axis="both", which="major", labelsize=8)
        g1.tick_params(axis="both", which="minor", labelsize=8)
        g1.set_title(value[0], fontweight="bold")
        if index == (len(values) - 1):
            for i in range(index + 1, len(axs)):
                axs[i].axis("off")

    plt.subplots_adjust(
        left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.55, hspace=0.65
    )
    # plt.show()
    plt.savefig("figures/conf_matrix_" + dataset_name + "_customised.png", dpi=200)
    plt.clf()


accuracy_list = list()
conf_matrices = list()
my_callbacks = [EarlyStopping(patience=3, monitor="loss", mode="min")]

cv = KFold(n_splits=10, random_state=42, shuffle=True)

# Neural network
evaluator = KerasClassifier(build_fn=create_model, epochs=100, batch_size=32)
results = cross_val_score(evaluator, X_text, y_values, cv=cv)
result = "NN: %.3f (%.3f)" % (results.mean(), results.std())
accuracy_list.append(result)
print(result)

if generate_figures:
    y_pred = cross_val_predict(evaluator, X_text, y_values, cv=cv)
    conf_mat = confusion_matrix(y_values, y_pred, normalize="true")
    conf_matrices.append(["Sieć neuronowa", conf_mat])

# SVM
model = svm.SVC()
scores = cross_val_score(model, X_text, y_values, scoring="accuracy", cv=cv, n_jobs=-1)
result = "SVM: %.3f (%.3f)" % (scores.mean(), scores.std())
accuracy_list.append(result)
print(result)

if generate_figures:
    y_pred = cross_val_predict(model, X_text, y_values, cv=cv)
    conf_mat = confusion_matrix(y_values, y_pred, normalize="true")
    conf_matrices.append(["Maszyna Wektorów Nośnych", conf_mat])

# Logistic Regression
model = LogisticRegression(solver="lbfgs", max_iter=1000)
scores = cross_val_score(model, X_text, y_values, scoring="accuracy", cv=cv, n_jobs=-1)
result = "LogisticRegression: %.3f (%.3f)" % (scores.mean(), scores.std())
accuracy_list.append(result)
print(result)

if generate_figures:
    y_pred = cross_val_predict(model, X_text, y_values, cv=cv)
    conf_mat = confusion_matrix(y_values, y_pred, normalize="true")
    conf_matrices.append(["Regresja Logistyczna", conf_mat])

# Random Forest
model = RandomForestClassifier()
scores = cross_val_score(model, X_text, y_values, scoring="accuracy", cv=cv, n_jobs=-1)
result = "RandomForest: %.3f (%.3f)" % (scores.mean(), scores.std())
accuracy_list.append(result)
print(result)

if generate_figures:
    y_pred = cross_val_predict(model, X_text, y_values, cv=cv)
    conf_mat = confusion_matrix(y_values, y_pred, normalize="true")
    conf_matrices.append(["Las Losowy", conf_mat])

# Gaussian Naive-Bayes
model = GaussianNB()
scores = cross_val_score(model, X_text, y_values, scoring="accuracy", cv=cv, n_jobs=-1)
result = "GaussianNB: %.3f (%.3f)" % (scores.mean(), scores.std())
accuracy_list.append(result)
print(result)

if generate_figures:
    y_pred = cross_val_predict(model, X_text, y_values, cv=cv)
    conf_mat = confusion_matrix(y_values, y_pred, normalize="true")
    conf_matrices.append(["Naiwny Bayes", conf_mat])

# AdaBoost
model = AdaBoostClassifier(n_estimators=800, random_state=42)
scores = cross_val_score(model, X_text, y_values, scoring="accuracy", cv=cv, n_jobs=-1)
result = "Ada: %.3f (%.3f)" % (scores.mean(), scores.std())
accuracy_list.append(result)
print(result)

if generate_figures:
    y_pred = cross_val_predict(model, X_text, y_values, cv=cv)
    conf_mat = confusion_matrix(y_values, y_pred, normalize="true")
    conf_matrices.append(["AdaBoost", conf_mat])

# KNN 3
model = KNeighborsClassifier(3)
scores = cross_val_score(model, X_text, y_values, scoring="accuracy", cv=cv, n_jobs=-1)
result = "KNN3: %.3f (%.3f)" % (scores.mean(), scores.std())
accuracy_list.append(result)
print(result)

if generate_figures:
    y_pred = cross_val_predict(model, X_text, y_values, cv=cv)
    conf_mat = confusion_matrix(y_values, y_pred, normalize="true")
    conf_matrices.append(["KNN 3", conf_mat])

# KNN 5
model = KNeighborsClassifier(5)
scores = cross_val_score(model, X_text, y_values, scoring="accuracy", cv=cv, n_jobs=-1)
result = "KNN5: %.3f (%.3f)" % (scores.mean(), scores.std())
accuracy_list.append(result)
print(result)

if generate_figures:
    y_pred = cross_val_predict(model, X_text, y_values, cv=cv)
    conf_mat = confusion_matrix(y_values, y_pred, normalize="true")
    conf_matrices.append(["KNN 5", conf_mat])

# KNN 7
model = KNeighborsClassifier(7)
scores = cross_val_score(model, X_text, y_values, scoring="accuracy", cv=cv, n_jobs=-1)
result = "KNN7: %.3f (%.3f)" % (scores.mean(), scores.std())
accuracy_list.append(result)
print(result)

file1 = open("figures/acc_customized_" + dataset_name + ".txt", "w")
file1.writelines("\n".join(accuracy_list))
file1.close()

if generate_figures:
    y_pred = cross_val_predict(model, X_text, y_values, cv=cv)
    conf_mat = confusion_matrix(y_values, y_pred, normalize="true")
    conf_matrices.append(["KNN 7", conf_mat])

if generate_figures:
    generate_heatmaps(conf_matrices)

    model = RandomForestClassifier()
    model.fit(X_text, y_values)

    shap.initjs()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_text)

    plt.subplots_adjust(
        left=0.3, bottom=0.1, right=0.9, top=0.95, wspace=0.8, hspace=0.9
    )
    shap.summary_plot(
        shap_values,
        features=X_text,
        feature_names=columns_polish,
        class_names=["Nie-plotka", "Plotka"],
        plot_type="bar",
        show=False,
    )
    plt.savefig("figures/features_" + dataset_name + ".png", dpi=200)

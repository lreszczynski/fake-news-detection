import re
import sys
from pathlib import Path
from unicodedata import category

import gensim
import nltk
import pandas as pd
import seaborn as sn
from gensim import downloader
from keras.wrappers.scikit_learn import KerasClassifier
from matplotlib import pyplot as plt
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers.core import Dense, Dropout
from tensorflow.python.keras.models import Sequential

pd.options.mode.chained_assignment = None

nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")
nltk.download("vader_lexicon")
nltk.download("stopwords")
nltk.download("punkt")

chrs = (chr(i) for i in range(sys.maxunicode + 1))
punctuation = set(c for c in chrs if category(c).startswith("P"))

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))
sentiment_analyzer = SentimentIntensityAnalyzer()

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

for i, s in enumerate(text):
    parsed = re.sub(
        r"http\S+", "", s
    )  # to remove links that start with HTTP/HTTPS in the tweet
    parsed = re.sub(r"@\w+|#", "", parsed)  # Remove user @ references and ‘#’ from text
    word_tokens = word_tokenize(parsed)
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
    filtered_sentence = [lemmatizer.lemmatize(s) for s in filtered_sentence]
    # filtered_sentence = [stemmer.stem(s) for s in filtered_sentence]
    filtered_sentence = [w for w in filtered_sentence if w not in punctuation]
    text[i] = filtered_sentence

embeddings = gensim.downloader.load("glove-twitter-100")

docs_vectors = pd.DataFrame()
for doc in df["text"]:
    temp = pd.DataFrame()
    for word in doc:
        try:
            word_vec = embeddings[word]
            temp = temp.append(pd.Series(word_vec), ignore_index=True)
        except:
            pass
    doc_vector = temp.mean()  # take the average of each column(w0, w1, w2,........w300)
    docs_vectors = docs_vectors.append(
        doc_vector, ignore_index=True
    )  # append each document value to the final df

docs_vectors["target"] = df["target"]
docs_vectors = docs_vectors.dropna()
X_text = docs_vectors.drop("target", axis=1)
y_values = docs_vectors["target"]


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
    plt.savefig("figures/conf_matrix_" + dataset_name + "_w2v.png", dpi=200)
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

if generate_figures:
    y_pred = cross_val_predict(model, X_text, y_values, cv=cv)
    conf_mat = confusion_matrix(y_values, y_pred, normalize="true")
    conf_matrices.append(["KNN 7", conf_mat])

file1 = open("figures/acc_w2v_" + dataset_name + ".txt", "w")
file1.writelines("\n".join(accuracy_list))
file1.close()

if generate_figures:
    generate_heatmaps(conf_matrices)

import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB

dataset_path = 'spam.csv'

if not os.path.exists(dataset_path):
    raise FileNotFoundError(
        f"Dataset not found at '{dataset_path}'. Please place the file in the script directory."
    )

if dataset_path.lower().endswith(('.xlsx', '.xls')):
    df = pd.read_excel(dataset_path)
else:
    df = pd.read_csv(dataset_path, encoding='latin-1')

if 'label' in df.columns and 'message' in df.columns:
    pass
else:
    df = df.rename(columns={df.columns[0]: 'label', df.columns[1]: 'message'})

df.dropna(inplace=True)

def extract_features(text):
    text = str(text)
    length = len(text)
    word_count = len(text.split())
    capitals = sum(1 for c in text if c.isupper())
    punctuation = sum(1 for c in text if c in "!?.,;:")
    contains_link = 1 if ("http" in text or "www" in text) else 0
    contains_number = 1 if any(c.isdigit() for c in text) else 0
    spam_keywords = ["free", "win", "offer", "urgent", "prize", "credit", "claim"]
    contains_keyword = 1 if any(word in text.lower() for word in spam_keywords) else 0
    return [length, word_count, capitals, punctuation, contains_link, contains_number, contains_keyword]

features = df["message"].apply(extract_features)
X = np.array(list(features))

df['label'] = df['label'].map({'ham': 0, 'spam': 1})
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# decision tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# multinomial naive bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)


dt_pred = dt_model.predict(X_test)
nb_pred = nb_model.predict(X_test)

# evaluation
def evaluate_model(name, y_true, y_pred):
    print(f"\n==== {name} ====")
    print("Accuracy:", round(accuracy_score(y_true, y_pred) * 100, 2), "%")
    print("Precision:", round(precision_score(y_true, y_pred) * 100, 2), "%")
    print("Recall:", round(recall_score(y_true, y_pred) * 100, 2), "%")
    print("F1 Score:", round(f1_score(y_true, y_pred) * 100, 2), "%")
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

# show results
evaluate_model("Decision Tree Classifier", y_test, dt_pred)
evaluate_model("Multinomial Naive Bayes", y_test, nb_pred)


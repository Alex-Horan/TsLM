import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from cog.torque import Graph
import pandas as pd
import matplotlib.pyplot as plt
import stanza

# nlp = stanza.Pipeline("en", download_method="reuse_resources")

text = pd.read_csv("intent_training.csv", usecols=['text'])
labels = pd.read_csv("intent_training.csv", usecols=["label"])
text = text.values.flatten().tolist()
labels = labels.values.flatten().tolist()

unseen = [
    "Could you explain what a data warehouse is?",
    "In which region can the Amazon rainforest be found?",
    "What causes a rainbow to appear after rainfall?",
    "What steps does a compiler follow to convert source code?",
    "Can you tell me where the headquarters of the World Health Organization is?"
]

X_train, X_test, y_train, y_test = train_test_split(text, labels, test_size=0.25, random_state=67)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True)),
    ("nb", MultinomialNB())
])

params = {
    "nb__alpha": [0.01, 0.1, 0.3, 0.5, 1.0, 2.0],
    "tfidf__ngram_range": [(1, 1), (1, 2)],
}

#cross testing for best fit every time :3
grid = GridSearchCV(pipeline, params, cv=3, scoring="accuracy")
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))


df = pd.read_csv("graph_data.csv")


G = Graph()

def populateGraph(graph):
    for i in range(len(df)):
        pass


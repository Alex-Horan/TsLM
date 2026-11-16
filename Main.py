import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from cog.torque import Graph
import pandas as pd
import stanza
import re

nlp = stanza.Pipeline("en", processors="tokenize,pos,lemma,depparse", download_method="reuse_resources")
df = pd.read_csv("graph_data.csv", header=None, names=["a", "relation", "b"])


text = pd.read_csv("intent_training.csv", usecols=['text'])
labels = pd.read_csv("intent_training.csv", usecols=["label"])
text = text.values.flatten().tolist()
labels = labels.values.flatten().tolist()

# normalizes the format of strings for data
def norm(s):
    return re.sub(r"\s+", " ", str(s).replace("_", " ").strip().lower())

df = df.map(norm)


unseen = [
    "What is an apple?",
    "What is a cylinder a part of?",
    "what is the cause of lava flow?",
    "what is snow?",
    "where is Paris?"
]

#these words are skipped when going through the nlp pipeline because it just makes it take longer and adds no additional information
SKIP = {"what", "who", "when", "where", "why", "how", "you", "me", "i", "it"}

INTENT_RELATIONS = {
    "EXPLAIN":  ["is a"],
    "DESCRIBE": ["has property"],
    "LOCATION": ["part of"],
    "PROCESS":  ["causes"],
}


KNOWN_ENTITIES = set(df["a"]) | set(df["b"])



# --------------- DEFINITIONS -------------------------

G = Graph("new_kg")


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



def populateGraph(graph):
    for i in range(len(df)):
        a, r, b = df.iloc[i]["a"], df.iloc[i]["relation"], df.iloc[i]["b"]
        graph.put(a, r, b)

populateGraph(G)

# -------------------- Subject extracting -----------------------------------
def noun_phrase(sent, head):
    ids = {head.id} | {w.id for w in sent.words if w.head == head.id and w.deprel == "compound"}
    return " ".join(w.text.lower() for w in sent.words if w.id in ids)

def extract_subject(sentence):
    doc = nlp(sentence)
    sent = doc.sentences[0]
    candidates = []

    # priority: subject, passive subj, obj, oblique, then any noun it can find
    for deprels in (["nsubj"], ["nsubj:pass"], ["obj"], ["obl", "nmod"], None):
        for w in sent.words:
            if w.upos not in ("NOUN", "PROPN"):
                continue
            if w.text.lower() in SKIP:
                continue
            if deprels is None or w.deprel in deprels:
                candidates.append((noun_phrase(sent, w), w.lemma.lower()))
    # prefer pre-existing connections in the graph
    for phrase, lemma in candidates:
        for option in (phrase, lemma):
            if norm(option) in KNOWN_ENTITIES:
                return norm(option)
    return norm(candidates[0][0]) if candidates else None


# ------------------- Vectorizing process for the graphdb ----------------------

node_stats = G.vectorize()
rel_words = sorted({r for rels in INTENT_RELATIONS.values() for r in rels})
rel_stats = G.vectorize(rel_words)

def embed(text):
    """fetch a vector by exact string (normalized)"""
    v = G.get_embedding(norm(text))
    return np.array(v, dtype=float) if v is not None else None

def rel_embed(rel):
    return embed(rel)


# ---------------- DistMult ---------------------------
# yes i know this is bad but i need some form of scoring and this is easy
def distmult(a,r,b):
    return float(np.sum(a*r*b))

# ---------------- Answering -------------

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))

def answer(sentence, top_k=3):
    intent = best_model.predict([sentence])[0].upper()
    subject = extract_subject(sentence)
    relations = INTENT_RELATIONS.get(intent, [])
    print(f"\nQ: {sentence}\n   intent={intent}    subject={subject}    relations={relations}")

    if subject is None:
        print("    no subject found")
        return []

    a_vec = embed(subject)
    scored = []
    for rel in relations:
        r_vec = embed(rel)
        result = G.v(subject).out(rel).all()
        for item in result.get("result", []):
            b = item["id"]
            b_vec = embed(b)
            if a_vec is None or r_vec is None or b_vec is None:
                continue
            scored.append((subject, rel, b, distmult(a_vec, r_vec, b_vec)))

    scored.sort(key=lambda t: t[3], reverse=True)
    if not scored:
        print("    no matching triples")
    for s, rel, b, sc, in scored[:top_k]:
        print(f"  ({s}) -[{rel}]-> ({b})  |  score={sc:.4f}")
    return scored

for q in unseen:
    answer(q)
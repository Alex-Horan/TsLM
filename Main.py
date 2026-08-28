import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import spacy

nlp = spacy.load("en_core_web_md")

texts = [
    # EXPLAIN
    "why is the sky blue",
    "what makes the ocean salty",
    "ice floats on water for what reason",
    "the roman empire fell because of what",
    "leaves turn orange and red in autumn, why",
    "what causes tides to rise and fall",
    "magnets pull on metal because of what force",
    "goosebumps happen for what reason",
    "what makes bread rise in the oven",
    "stars appear to twinkle at night, what causes that",
    "world war one broke out for what reasons",
    "caffeine keeps people awake, how come",
    "cats purr for what reason",
    "metal rusting is caused by what",
    "earthquakes are triggered by what",
    "glass lets light pass through, why is that",
    "yawning when tired happens because of what",
    "helium changes the pitch of your voice, why",
    "birds fly south in winter for what reason",
    "soap cuts through grease because of what",
    "what triggers a volcano to erupt",
    "the sun looks red at sunset, what's the reason",
    "some animals hibernate all winter, why is that",
    "vinegar fizzes when mixed with baking soda, why",
    "what's the reason dreams happen during sleep",

    # DESCRIBE
    "tell me about apples",
    "gravity, what exactly is that",
    "give me a definition of entropy",
    "mitosis is a process, can you explain the concept",
    "paint me a picture of what a black hole is",
    "photosynthesis in plants, what's going on there",
    "a neuron in the brain, what is that",
    "gold as a material, what are its properties",
    "quantum mechanics is a branch of physics, describe it",
    "an ecosystem, what does that term actually mean",
    "picture a volcano for me, what does it look like",
    "algorithms in computing, what are those",
    "inflation as an economic term, define it",
    "a solar eclipse, what happens during one",
    "dna's building blocks, what are they made of",
    "enzymes in the body, what are they",
    "democracy as a system of government, what is it",
    "coral reefs, tell me what those are",
    "an atom's structure, break it down for me",
    "artificial intelligence, give me a rundown",
    "gross domestic product, what does that measure",
    "glaciers, what are they exactly",
    "tornadoes, describe what one looks like",
    "osmosis, what's the definition",
    "hurricanes, what are they",

    # LOCATION
    "paris is located where",
    "point me toward the eiffel tower",
    "mount everest sits in which country",
    "penguins live in what parts of the world",
    "is there a hospital nearby",
    "the great wall of china stretches through where",
    "the amazon rainforest covers which region",
    "silicon valley is situated where",
    "polar bears are found in which regions",
    "the sahara desert spans across where",
    "the statue of liberty stands where",
    "closest gas station to here",
    "kilimanjaro is a mountain located in which country",
    "kangaroos are native to what part of the world",
    "the grand canyon is carved into which state",
    "nearest pharmacy around here",
    "the taj mahal was built in which city",
    "dolphins are commonly found in which waters",
    "the dead sea borders which countries",
    "niagara falls sits between which two countries",
    "closest airport from this location",
    "pandas live in the wild in which country",
    "the colosseum still stands in which city",
    "antarctica is positioned where on the globe",
    "the nile river flows through which countries",

    # PROCESS
    "walk me through how a car engine runs",
    "plants make their own food, what's the process",
    "explain the steps behind how the internet works",
    "what keeps airplanes up in the sky",
    "vaccines protect the body, what's the mechanism",
    "a fridge keeps things cold, what's happening inside it",
    "steps for baking bread from scratch",
    "the heart pumps blood, what's the mechanism there",
    "solar panels turn sunlight into electricity, how",
    "show me the steps to tie a bowline knot",
    "washing machines get clothes clean, what's the process",
    "bees produce honey, walk me through how",
    "steps to swap out a flat tire",
    "wifi sends data through the air, how does that work",
    "brewing coffee with a french press, what are the steps",
    "batteries store energy, what's the process behind that",
    "setting up a new email account, what are the steps",
    "the water cycle moves water around the planet, explain how",
    "steps for starting a vegetable garden",
    "microwaves heat food, what's actually happening",
    "putting together flat pack furniture, what's the process",
    "electric motors convert electricity into motion, how",
    "installing python on a windows machine, what are the steps",
    "digestion breaks down food, walk me through the process",
    "folding a paper airplane, what are the steps",
]

labels = (
    ["EXPLAIN"] * 25 +
    ["DESCRIBE"] * 25 +
    ["LOCATION"] * 25 +
    ["PROCESS"] * 25
)


unseen = [
    "Could you explain what a data warehouse is?",
    "In which region can the Amazon rainforest be found?",
    "What causes a rainbow to appear after rainfall?",
    "What steps does a compiler follow to convert source code?",
    "Can you tell me where the headquarters of the World Health Organization is?"
]




X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.25, random_state=42,
)
    
    # turns sentences into importance scales using tfidf, uses this rather than plain count for better results
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True)),
    ("nb", MultinomialNB()),
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

# x = ["How does a bike work?"]
# y = best_model.predict(x)
# print(y)

df = pd.read_csv("graph_training_data.csv")

G = nx.MultiGraph()

def populateGraph(graph):
    for i in range(len(df)):
        graph.add_edge(df["A"][i], df["B"][i], weight=df['WEIGHT'][i], relation=df["RELATION"][i])

populateGraph(G)

# sp = dict(nx.all_pairs_all_shortest_paths(G))
# print(sp[3])
# print(list(G.nodes)[:15])
# print(G.is_multigraph())

# node_labels = nx.get_node_attributes(G, 'value')
# pos = nx.spring_layout(G)
# nx.draw(G, pos=pos, with_labels=True)
# plt.show()

def findConnection(G, data, intent=None):
    relations = []
    if not (data.get("subj") and data.get("obj") and data.get("comp")):
        return "unknown."
    if intent == "DESCRIBE":
        relations=["isA", "hasQuality", "contains"]
    elif intent == "EXPLAIN":
        relations=["causedBy"]
    elif intent == "LOCATION":
        relations = ["isIn", "hasInstance"]
    elif intent == "PROCESS":
        relations = ["hasStep"]
    else:
        return "Unknown intention"




def pipeLine(i: list):
    intent = best_model.predict(i)
    doc = nlp(i[0])
    keywords = [tok.text for tok in doc if tok.pos_ in ("NOUN", "PROPN", "ADJ") and not tok.is_stop]
    subject = ""
    obj = ""
    comp = ""

    for tok in doc:
        if tok.dep_ in ("nsubj", "nsubjpass"):
            subject = tok.text
            # print(subject)
        if tok.dep_ in ("dobj", "attr", "pobj"):
            obj = tok.text
            # print(obj)
        if tok.dep_ == "compound":
            comp = tok.text

    return {"subj": subject, "obj": obj, "comp": comp}

keyword = pipeLine(["what color is an apple?"])
# print(keyword)
# findConnection(G, "Fruit", "Apple")
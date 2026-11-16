import stanza
import concurrent.futures

def initWorker():
    global nlp

    nlp = stanza.Pipeline('en', download_method='reuse_resources')



sentences = []

with open("raw_sent.txt", "r") as f:
    content = f.read()
    sentences.append(content.split(","))

def processText(text):
    doc = nlp(text)
    return doc

def parallel(text_list):
    with concurrent.futures.ProcessPoolExecutor(max_workers=10, initializer=initWorker) as executor:
        result = list(executor.map(processText, text_list))
    return result

# sentences = [nlp(i) for i in sentences[0]] # run if you hate your cpu ig
# print(parallel(sentences[0]))
# def parse(sentences):

if __name__ == '__main__':
    print(parallel(sentences[0]))
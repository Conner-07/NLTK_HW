import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import spaCy


# Ensure NLTK resources are available
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load English NLP model
nlp = spaCy.load('en_core_web_sm')

# Load text files
file_paths = {
    "text1": "/mnt/data/RJ_Lovecraft.txt",
    "text2": "/mnt/data/RJ_Tolkein.txt",
    "text3": "/mnt/data/RJ_Martin.txt",
    "text4": "/mnt/data/Martin.txt"
}

texts = {}
for name, path in file_paths.items():
    with open(path, 'r') as file:
        texts[name] = file.read()

# Tokenization, Stemming, and Lemmatization
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def process_text(text):
    tokens = word_tokenize(text.lower())
    stems = [ps.stem(token) for token in tokens if token.isalpha()]
    lemmas = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha()]
    return tokens, stems, lemmas

processed_texts = {}
for name, text in texts.items():
    tokens, stems, lemmas = process_text(text)
    processed_texts[name] = {"tokens": tokens, "stems": stems, "lemmas": lemmas}

# Top 20 Tokens
top_tokens = {}
for name, data in processed_texts.items():
    counter = Counter(data["tokens"])
    top_tokens[name] = counter.most_common(20)

# Named Entity Recognition
named_entities = {}
for name, text in texts.items():
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents]
    named_entities[name] = len(entities)

# N-gram Analysis
def extract_ngrams(text, n=3):
    vectorizer = CountVectorizer(ngram_range=(n, n))
    ngram_counts = vectorizer.fit_transform([text])
    ngram_sum = ngram_counts.sum(axis=0)
    ngram_freq = [(word, ngram_sum[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    ngram_freq = sorted(ngram_freq, key=lambda x: x[1], reverse=True)
    return ngram_freq[:10]

ngrams = {}
for name, text in texts.items():
    ngrams[name] = extract_ngrams(text)

# Results
results = {
    "top_tokens": top_tokens,
    "named_entities": named_entities,
    "ngrams": ngrams
}

# Display Results
import pandas as pd
for key, value in results.items():
    print(f"---- {key.upper()} ----")
    for text_name, result in value.items():
        print(f"{text_name}: {result}")
    print("\n")

# Author Identification (Basic Similarity Comparison)
def compare_texts_by_ngrams(ngrams, text1, text2):
    ngrams1 = set([item[0] for item in ngrams[text1]])
    ngrams2 = set([item[0] for item in ngrams[text2]])
    common = ngrams1.intersection(ngrams2)
    return common

comparison_results = {
    "text4_vs_text1": compare_texts_by_ngrams(ngrams, "text4", "text1"),
    "text4_vs_text2": compare_texts_by_ngrams(ngrams, "text4", "text2"),
    "text4_vs_text3": compare_texts_by_ngrams(ngrams, "text4", "text3")
}

print("---- AUTHORSHIP ANALYSIS ----")
for comparison, common in comparison_results.items():
    print(f"{comparison}: {common}")


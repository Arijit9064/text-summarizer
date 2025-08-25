import nltk
import re
import heapq

# Download required resources for NLTK (only first time)
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

def summarize_text(text, num_sentences=3):
    # Preprocess text
    text = re.sub(r'\s+', ' ', text)
    
    # Tokenize into sentences
    sentences = sent_tokenize(text)
    
    # Tokenize into words
    words = word_tokenize(text.lower())
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w.isalnum() and w not in stop_words]
    
    # Word frequencies
    word_frequencies = {}
    for word in words:
        if word not in word_frequencies:
            word_frequencies[word] = 1
        else:
            word_frequencies[word] += 1
    
    # Normalize frequencies
    max_freq = max(word_frequencies.values())
    for word in word_frequencies:
        word_frequencies[word] = word_frequencies[word] / max_freq
    
    # Sentence scores
    sentence_scores = {}
    for sent in sentences:
        for word in word_tokenize(sent.lower()):
            if word in word_frequencies:
                if sent not in sentence_scores:
                    sentence_scores[sent] = word_frequencies[word]
                else:
                    sentence_scores[sent] += word_frequencies[word]
    
    # Select top sentences
    summary_sentences = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    summary = " ".join(summary_sentences)
    return summary


if __name__ == "__main__":
    # Read input text file
    with open("demo_input.txt", "r", encoding="utf-8") as f:
        article = f.read()
    
    print("\n=== ORIGINAL TEXT ===\n")
    print(article[:500] + "...")  # print first 500 chars only
    
    print("\n=== SUMMARY ===\n")
    summary = summarize_text(article, num_sentences=3)
    print(summary)

from dictionary import training_data
import sys
import nltk 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

n = len(sys.argv)
ps = PorterStemmer()

if (n != 2):
    print("Error: Not enough or too many input arguments.", file=sys.stderr)
    exit()

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
training = training_data('DisneylandReviews.csv')

def stopwords(sentence):
    filtered_sentence = [w for w in sentence.split() if not w.lower() in stop_words]
    return ' '.join(filtered_sentence)

def stemmer(sentence):
    filtered_sentence = [ps.stem(w) for w in sentence.split()]
    return ' '.join(filtered_sentence)

def lowercasing(sentence):
    return sentence.lower()

def preprocessingNoSkip(data):
    for sentence in data.values():
        stopwords(stemmer(lowercasing(sentence)))
    return

def preprocessingSkip(data):
    for sentence in data.values():
        stopwords(lowercasing(sentence))
    return

if (sys.argv[1].upper() == "YES"):
    preprocessingSkip(training)
elif (sys.argv[1].upper() == "NO"):
    preprocessingNoSkip(training)
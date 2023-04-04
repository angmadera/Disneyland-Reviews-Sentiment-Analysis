from dictionary import data
import nltk 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

ps = PorterStemmer()

x = input("Would you like to skip the stemming preprocessing step? (YES/NO) ")
print(x)

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
training, test = data('DisneylandReviews.csv')
frequency = {}
wordCount = {'1':0, '2':0, '3':0, '4':0, '5':0}
reviewDict = {'1':0, '2':0, '3':0, '4':0, '5':0}

def vocabulary(arr):
    rating = arr[0]
    sentence = arr[1]
    for w in sentence:
        if w in frequency.keys():
            labels = frequency[w]
            labels[int(rating) - 1] = labels[int(rating) - 1] + 1
            wordCount.update({rating: wordCount[rating] + 1})
            labels[5] = labels[5] + 1
            frequency.update({w: labels})
        else:
            labels = [1, 1, 1, 1, 1, 5]
            labels[int(rating) - 1] = labels[int(rating) - 1] + 1
            labels[5] = labels[5] + 1
            wordCount.update({rating: wordCount[rating] + 2})
            frequency.update({w: labels})
    return

def tokenize(sentence):
    pattern = r"""(?x)                   # set flag to allow verbose regexps
              (?:[A-Z]\.)+           # abbreviations, e.g. U.S.A.
              |\d+(?:\.\d+)?%?       # numbers, incl. currency and percentages
              |\w+(?:[-']\w+)*       # words w/ optional internal hyphens/apostrophe
              |(?:[+/\-@&*])         # special characters with meanings
            """
    tokenizer = nltk.tokenize.regexp.RegexpTokenizer(pattern)
    tokens = tokenizer.tokenize(sentence)
    return tokens

def preprocessing(data):
    reviewCount = 0
    for review in data.values():
        reviewCount = reviewCount + 1
        filtered_review = [ps.stem(w.lower()) for w in set(tokenize(review[1])) if not ps.stem(w.lower()) in stop_words]
        finalReview = set(filtered_review)
        reviewDict.update({review[0]: reviewDict[review[0]] + 1})
        vocabulary([review[0], filtered_review])
    return reviewCount

def preprocessingSkip(data):
    reviewCount = 0
    for review in data.values():
        reviewCount = reviewCount + 1
        filtered_review = [w.lower() for w in tokenize(review[1]) if not w.lower() in stop_words]
        finalReview = set(filtered_review)
        vocabulary([review[0], finalReview])
        reviewDict.update({review[0]: reviewDict[review[0]] + 1})
    return reviewCount

if (x.upper() == "YES"):
    preprocessingSkip(training)
elif (x.upper() == "NO"):
    preprocessing(training)

# print(frequency['liberate'][5])
# print(wordCount)
# print(reviewDict)

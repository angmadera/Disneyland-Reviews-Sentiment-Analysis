from dictionary import data
import nltk 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import math
import numpy

ps = PorterStemmer()

x = input("Would you like to skip the stemming preprocessing step? (YES/NO) ")

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
training_data, test_data = data('DisneylandReviews.csv')
revCount = 0
frequency = {}
wordCount = {'1':0, '2':0, '3':0, '4':0, '5':0}
reviewDict = {'1':0, '2':0, '3':0, '4':0, '5':0}

#each word is a key, the value is an array containing frequency per category and total frequency (+1 for smoothing)
def vocabulary(arr):
    rating = arr[0]
    sentence = arr[1]
    for w in sentence:
        #if word already exists, update key to add one to the total frequency count and also to the corresponding label
        #also add one to word count
        if w in frequency.keys():
            labels = frequency[w]
            labels[int(rating) - 1] = labels[int(rating) - 1] + 1
            wordCount.update({rating: wordCount[rating] + 1})
            labels[5] = labels[5] + 1
            frequency.update({w: labels})
        else:
            #if word does not exist, add it to dictionary with initial value, then add one to the total frequency count and also to the corresponding label
            #add two to word count (+1 smoothing)
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

def preprocessing(review):
    if (x.upper() == "YES"):
        filtered_review = [ps.stem(w.lower()) for w in (tokenize(review)) if not ps.stem(w.lower()) in stop_words]
    elif (x.upper() == "NO"):
        filtered_review = [w.lower() for w in tokenize(review) if not ps.stem(w.lower()) in stop_words]
    return filtered_review

def preprocessingTraining(data):
    reviewCount = 0
    for review in data.values():
        #keep count of total # of reviews
        reviewCount = reviewCount + 1
        #updating dictionary that keeps count of number of total reviews per category ex. adds count to category '5' 
        reviewDict.update({review[0]: reviewDict[review[0]] + 1})
        #preprocessing step
        filtered_review = preprocessing(review[1])
        #removing repeating words
        finalReview = set(filtered_review)
        #adding to vocab of words
        vocabulary([review[0], finalReview])
    return reviewCount

revCount = preprocessingTraining(training_data)

def learning(label):
    #ex. total number of reviews labeled y / total number of reviews
    return reviewDict[label]/revCount

def learningProbability(word, label):
    #ex. total number of instances of word in review label y / total number of words in label y
    return frequency[word][int(label) - 1]/wordCount[label]

#ex. P(label = 1) * P(x1=word | label =1) * P(x2=word | label =1) *  P(x3=word | label =1) *  P(x4=word | label =1)
def probability(sentence, label):
    prob = []
    #finds P(label = 1)
    prob.append(learning(label))
    #finds P(x1=word | label =1), P(x2=word | label =1),  P(x3=word | label =1),  P(x4=word | label =1)
    for word in sentence:
        if word in frequency.keys():
            prob.append(learningProbability(word, label))
        else:
            continue
    #multiplies probability
    finalProb = numpy.prod(prob)
    return finalProb

#NOTE: remove print statements, they were just used to debug
def test(data):
    for review in data.values():
        prob = []
        filtered_sentence = preprocessing(review[1])
        print(filtered_sentence)
        #finds probability for labels 1 - 5
        for x in range(1, 6):
            prob.append(probability(filtered_sentence, str(x)))
        print(prob)
        #finds max probability and chooses that label
        maxIdx = prob.index(max(prob))
        print(maxIdx)
        chosenLabel = str(int(maxIdx) + 1)
        print(review[0], chosenLabel)
    return

test(test_data)
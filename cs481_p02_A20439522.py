from dictionary import data
import nltk 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import math
import numpy

ps = PorterStemmer()

nltk.download('stopwords')

print()
x = input("Would you like to skip the stemming preprocessing step? (YES/NO) ")

print()
print("Madera, Angelica, A20439552 solution:")

if (x.upper() == "NO"):
    print("Ignored pre-processing step: NONE")
elif (x.upper() == "YES"):
    print("Ignored pre-processing step: STEMMING")  
print()

stop_words = set(stopwords.words('english'))
training_data, test_data = data('DisneylandReviews.csv')
revCount = 0
frequency = {}
wordCount = {'1':0, '2':0, '3':0, '4':0, '5':0}
reviewDict = {'1':0, '2':0, '3':0, '4':0, '5':0}
rows, cols = (5, 5)
confusionMatrix = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
classifications = {'1': [0, 0, 0, 0], '2': [0, 0, 0, 0], '3': [0, 0, 0, 0], '4': [0, 0, 0, 0], '5': [0, 0, 0, 0]} #tp, fn, fp, tn

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

def preprocessing(review):
    if (x.upper() == "NO"):
        filtered_review = [ps.stem(w.lower()) for w in (tokenize(review)) if not ps.stem(w.lower()) in stop_words]
    elif (x.upper() == "YES"):
        filtered_review = [w.lower() for w in tokenize(review) if not w.lower() in stop_words]
    return filtered_review

def preprocessingTraining(data):
    reviewCount = 0
    for review in data.values():
        reviewCount = reviewCount + 1
        reviewDict.update({review[0]: reviewDict[review[0]] + 1})
        filtered_review = preprocessing(review[1])
        finalReview = set(filtered_review)
        vocabulary([review[0], finalReview])
    return reviewCount

print("Training classifier…")
revCount = preprocessingTraining(training_data)

def learning(label):
    return reviewDict[label]/revCount

def learningProbability(word, label):
    return frequency[word][int(label) - 1]/wordCount[label]

def probability(sentence, label):
    prob = []
    prob.append(learning(label))
    for word in sentence:
        if word in frequency.keys():
            prob.append(learningProbability(word, label))
        else:
            continue
    finalProb = numpy.prod(prob)
    return finalProb

def test(data):
    for review in data.values():
        prob = []
        filtered_sentence = preprocessing(review[1])
        for x in range(1, 6):
            prob.append(probability(filtered_sentence, str(x)))
        maxIdx = prob.index(max(prob))
        chosenLabel = str(int(maxIdx) + 1)
        row = confusionMatrix[int(review[0]) - 1]
        row[int(chosenLabel) - 1] = row[int(chosenLabel) - 1] + 1
        confusionMatrix[int(review[0]) - 1] = row
    return

def metricClassification(matrix):
    countTN = 0
    for x in range(5):    
        for y in range(5):
            if x == y:
                label = classifications[str(x + 1)]
                label[0] = label[0] + matrix[x][y]
                classifications.update({str(x + 1): label})
            else:
                label = classifications[str(x + 1)]
                label[1] = label[1] + matrix[x][y]
                label[2] = label[2] + matrix[y][x]
                classifications.update({str(x + 1): label})
                print(matrix[y][x])
                countTN += matrix[x][y]
    for x in range(5):
        label = classifications[str(x + 1)]
        label[3] = label[3] + countTN
        classifications.update({str(x + 1): label})
    return

    
print("Testing classifier…")
test(test_data)

print("Test results / metrics:")
print(confusionMatrix)
metricClassification(confusionMatrix)
print(classifications)
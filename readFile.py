import re
import os 

#script adapted from: https://www.kaggle.com/venkatramani/imdb-datascraper

interp = re.compile('[.;:!\'?~,\"()\[\]]')
space = re.compile('<br\s*/><br\s*/>|(\-)|(\/)')

def preprocess_reviews(reviews):
    reviews_train = []
    for line in open(reviews, 'r'):
        reviews_train.append(line.strip())
    for line in reviews_train:
        #replace shortenings by full words or delete
        line = re.sub('n\'t', ' not', line)
        line = re.sub('\'ll', '', line)
        line = re.sub('\'m', '', line)
        line = re.sub('\'s', '', line)
        line = re.sub('\'ve', '', line)
        line = re.sub('\'d', '', line)
        #delete articles
        line = re.sub(' a ', ' ', line)
        line = re.sub(' an ', ' ', line)
        line = re.sub(' the ', ' ', line)
    #delete interpunction and change all words to lower case
    reviews_train = [interp.sub("", line.lower()) for line in reviews_train]
    #delete addidional spaces
    reviews_train = [space.sub(" ", line) for line in reviews_train]
    return reviews_train
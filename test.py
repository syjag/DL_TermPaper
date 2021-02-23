import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from embed import embed_review
from readFile import preprocess_reviews

from main import code, model, kmeans

from collections import Counter
import os

TT = torch.TensorType

print("embedding positive")
X =[]
directory = 'aclImdb/test/pos1'
for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        reviews_train = preprocess_reviews(directory+'/'+filename)
        embed = embed_review(reviews_train)
        X.append(embed)
print("finished")

padded = pad_sequence(X, batch_first=True) #batch_fisrt=True am sonsten reihenefolge anders

code = model.encoder(padded)
dim_red = code.view(len(code),-1)
y_pred = kmeans.predict(dim_red.detach())

pos_clusters_dict = dict(Counter(y_pred))
if pos_clusters_dict[0] > pos_clusters_dict[1]:
    print("positive label: 0")
    accuracy = pos_clusters_dict[0]/len(y_pred)
    print("accuracy: ", accuracy)
else:
    print("positive label: 1")
    accuracy = pos_clusters_dict[1]/len(y_pred)
    print("accuracy: ", accuracy)

print('----------------------------------------------------------')

print("embedding negative")
X =[]
directory = 'aclImdb/test/neg1'
for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        reviews_train = preprocess_reviews(directory+'/'+filename)
        embed = embed_review(reviews_train)
        X.append(embed)
print("finished")

padded = pad_sequence(X, batch_first=True) #batch_fisrt=True am sonsten reihenefolge anders

code = model.encoder(padded)
dim_red = code.view(len(code),-1)
y_pred = kmeans.predict(dim_red.detach())

neg_clusters_dict = dict(Counter(y_pred))
if pos_clusters_dict[0] > pos_clusters_dict[1]:
    print("negative label: 1")
    accuracy = neg_clusters_dict[1]/len(y_pred)
    print("accuracy: ", accuracy)
else:
    print("negative label: 0")
    accuracy = neg_clusters_dict[0]/len(y_pred)
    print("accuracy: ", accuracy)
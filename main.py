import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from autoencoder import AE
from training import train
from embed import embed_review
from readFile import preprocess_reviews

from sklearn.cluster import KMeans
import os

TT = torch.TensorType

#Embedding of the reviews:
#first the reviews are preprocessed, than embeded, than each embedding is appended to the list X
print("embedding")
X =[]
directory = 'aclImdb/train/unsup1'
for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        reviews_train = preprocess_reviews(directory+'/'+filename)
        embed = embed_review(reviews_train)
        X.append(embed)
print("finished")

#the list has to be padded, so that every review has the same dimension
#reviews with under 300 words will be padded with zeroes
padded = pad_sequence(X, batch_first=True) #batch_fisrt=True to keep the order

#initialising of the autoencoder that will reduce the last dimension to 50 during the training
model = AE(200,100,50) 
loss = nn.MSELoss()

train(model, padded, loss)

#extracting the sparse embeddings after the training
code = model.encoder(padded)

#training the clusters on the pre-trained embeddings
kmeans = KMeans(n_clusters=2, random_state=25)
dim_red = code.view(len(code),-1)
y_pred = kmeans.fit_predict(dim_red.detach())
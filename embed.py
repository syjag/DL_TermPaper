import torch

from word_embedding import AtomicEmbedder

def embed_review(review):
    review_list = str(review).split()
    word_emb = AtomicEmbedder(review_list, 200)
    embs = []
    #not many reviews have over 300 words, so to prevent too many zeroes, just stop at 300
    if len(review_list)<=300:
        for i in range(len(review_list)):
            embs.append(word_emb.forward(review_list[i]))
    else:
        for i in range(300):
            embs.append(word_emb.forward(review_list[i]))

    review_emb = torch.stack(embs)

    return review_emb
#script from the lecture

from typing import Iterable, Set

from abc import ABC, abstractmethod
import io

import torch
import torch.nn as nn

from embedding import Embedding

TT = torch.TensorType
Word = str

class WordEmbedder(ABC, nn.Module):

    @abstractmethod
    def forward(self, word: Word) -> TT:
        """Embed the given word."""
        pass

    # @abstractmethod
    def forwards(self, words: Iterable[Word]) -> TT:
        """Embed the given words."""
        # Default implementation.  Re-implement for speed
        # in a sub-class.
        return torch.stack([self.forward(word) for word in words])

    @abstractmethod
    def embedding_size(self) -> int:
        """Return the size of the embedding vectors."""
        pass

class AtomicEmbedder(WordEmbedder):

    def __init__(self, vocab: Set[Word], emb_size: int,
                 case_insensitive=False):
        """Create the word embedder for the given vocabulary.
        Arguments:
            vocab: vocabulary of words to embed
            emb_size: the size of embedding vectors
            case_insensitive: should the embedder be case-insensitive?
        """
        # The following line is required in each custom neural Module.
        super(AtomicEmbedder, self).__init__()
        # Keep info about the case sensitivity
        self.case_insensitive = case_insensitive
        # Calculate the modified vocabulary
        vocab = set(self.preprocess(x) for x in vocab)
        # Initialize the generic embedding module
        self.emb = Embedding(vocab, emb_size)

    def preprocess(self, word: Word) -> Word:
        """Preprocessing function"""
        # We use word.lower() to make the embedder case-insensitive
        if self.case_insensitive:
            return word.lower()
        else:
            return word

    def forward(self, word: Word) -> TT:
        """Embed the given word as a vector."""
        return self.emb(self.preprocess(word))

    def forwards(self, words: Iterable[Word]) -> TT:
        """Embed the given sequence of words."""
        # This is faster than the default implementation (see the
        # `WordEmbedder` class) because it relies on the more
        # performant `self.emb.fowards` method.
        return self.emb.forwards(map(self.preprocess, words))

    def embedding_size(self) -> int:
        """Return the embedding size of the word embedder."""
        return self.emb.embedding_size()
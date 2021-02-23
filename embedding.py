#from the lecture script
from typing import Iterable, Dict

import torch
import torch.nn as nn

TT = torch.TensorType

class Embedding(nn.Module):

    def __init__(self, alphabet: set, emb_size: int):

        # The following line is required in nn.Module subclasses
        super(Embedding, self).__init__()
        # Keep the embedding size
        self.emb_size = emb_size
        # Create the mapping from alphabet/vocabulary to indices
        self.obj_to_ix = {}  # type: Dict
        for ix, obj in enumerate(alphabet):
            self.obj_to_ix[obj] = ix
        # We use `len(self.obj_to_ix)` as a padding index, i.e.,
        # the index of the embedding vector fixed to 0 and which
        # is used to represent out-of-vocabulary words.
        self.padding_idx = len(self.obj_to_ix)
        # Create the nn.Embedding module; the vocabulary size
        # is set to `len(self.obj_to_ix)+1` because of the padding
        # index.
        self.emb = nn.Embedding(
            len(self.obj_to_ix)+1,
            emb_size,
            padding_idx=self.padding_idx
        )

    def embedding_size(self) -> int:
        """Return the embedding size."""
        return self.emb_size

    def forward(self, sym) -> TT:
        """Embed the given symbol."""
        try:
            ix = self.obj_to_ix[sym]
        except KeyError:
            # In case of out-of-vocabulary symbol/word,
            # use the padding index
            ix = self.padding_idx
        return self.emb(torch.tensor(ix, dtype=torch.long))

    def forwards(self, syms: Iterable) -> TT:
        """Embed the given sequence of symbols (word)."""
        ixs = []
        for sym in syms:
            try:
                ixs.append(self.obj_to_ix[sym])
            except KeyError:
                ixs.append(self.padding_idx)
        return self.emb(torch.LongTensor(ixs))
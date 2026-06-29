"""
Code reused in scripts
"""

import torch
import numpy as np
import random
import torchhd

def set_random_seed(seed=0):
    """Set random seed to favor reproducible experiments"""
    # Set random seed
    # These seed setings were did not change the accuracy, but the pytorch
    # documentation recommends setting them:
    # https://pytorch.org/docs/stable/notes/randomness.html
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def make_random_queries(codebooks, num_queries):
    """
    Create random query vectors from the codebooks. Returns the query vectors
    and the codebooks' indices to build them.
    """
    F, M, D = codebooks.shape
    B = num_queries

    # Expand codebooks to a read only variable to add a batch size
    # dimension, cb.shape = [Batch, Features, Codebook Size, Dim].
    cb = codebooks.unsqueeze(0).expand(B, -1, -1, -1)
    # Index one vector from each codebook #
    # Create random indices to create query vectors.
    inds_rand = torch.randint(M, (B, F,))
    inds = inds_rand.view(B, F, 1, 1).expand(-1, -1, 1, D)

    # Retrieve a batch of factors from codebooks
    vectors = cb.gather(-2, inds).squeeze(-2) # vectors = [Batch, Features, Dim]

    s = torchhd.multibind(vectors) # s = [Batch, Dim]

    return s, inds_rand


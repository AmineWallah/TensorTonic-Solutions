import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    max_len = max(len(seq) for seq in seqs) if max_len is None else max_len
        
    padded = np.full((len(seqs), max_len), pad_value)
    
    for i, seq in enumerate(seqs):
        for j, val in enumerate(seq[:max_len]):
            padded[i,j] = val

    return padded

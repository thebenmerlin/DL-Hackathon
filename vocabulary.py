import pickle
import os

class Vocabulary:
    """Vocabulary class for mapping words to indices and vice versa."""
    
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        
    def add_word(self, word):
        """Add a word to the vocabulary."""
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
    
    def __call__(self, word):
        """Get the index for a word."""
        return self.word2idx.get(word, self.word2idx.get('<unk>', 0))
    
    def __len__(self):
        """Get the vocabulary size."""
        return len(self.word2idx)
    
    def save(self, path):
        """Save vocabulary to file."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path):
        """Load vocabulary from file."""
        with open(path, 'rb') as f:
            vocab = pickle.load(f)
        return vocab


def build_vocabulary(captions, min_freq=5):
    """
    Build vocabulary from captions.
    Args:
        captions: List of tokenized captions (list of lists of words)
        min_freq: Minimum frequency for a word to be included
    Returns:
        vocab: Vocabulary object
    """
    word_freq = {}
    
    # Count word frequencies
    for caption in captions:
        for word in caption:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Create vocabulary
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')
    
    for word, freq in word_freq.items():
        if freq >= min_freq:
            vocab.add_word(word)
    
    return vocab

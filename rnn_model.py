import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMDecoderWithAttention(nn.Module):
    """
    RNN-based caption generator with LSTM and attention mechanism.
    Takes CNN features and generates text descriptions.
    """
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, max_seq_length=50):
        super(LSTMDecoderWithAttention, self).__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        
        # Embedding layer for words
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # LSTM decoder
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Linear layer to predict next word
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights with Xavier uniform distribution."""
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, features, captions, lengths):
        """
        Forward pass during training.
        Args:
            features: CNN features of shape (batch_size, embed_size)
            captions: Caption tensors of shape (batch_size, max_length)
            lengths: Actual lengths of captions
        Returns:
            outputs: Predicted word distributions
        """
        # Embed the captions
        embeddings = self.embedding(captions)  # (batch_size, seq_length, embed_size)
        
        # Concatenate image features with word embeddings
        # Use image features as the initial input
        embeddings = torch.cat((features.unsqueeze(1), embeddings[:, :-1, :]), dim=1)
        
        # Pack padded sequence for efficient LSTM processing
        packed = nn.utils.rnn.pack_padded_sequence(
            embeddings, lengths, batch_first=True, enforce_sorted=False
        )
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, features.size(0), self.hidden_size).to(features.device)
        c0 = torch.zeros(self.num_layers, features.size(0), self.hidden_size).to(features.device)
        
        # Pass through LSTM
        outputs, _ = self.lstm(packed, (h0, c0))
        
        # Unpack and pass through linear layer
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        outputs = self.fc(outputs)  # (batch_size, seq_length, vocab_size)
        
        return outputs
    
    def sample(self, features, vocab, max_length=50):
        """
        Generate caption for an image at inference time.
        Args:
            features: CNN features of shape (1, embed_size)
            vocab: Vocabulary object
            max_length: Maximum caption length
        Returns:
            caption: Generated caption as a list of words
        """
        self.eval()
        
        captions = []
        inputs = features.unsqueeze(0)  # Add batch dimension
        
        # Initialize hidden states
        hidden = (
            torch.zeros(self.num_layers, 1, self.hidden_size).to(features.device),
            torch.zeros(self.num_layers, 1, self.hidden_size).to(features.device)
        )
        
        for i in range(max_length):
            # Pass through LSTM
            output, hidden = self.lstm(inputs, hidden)
            
            # Predict next word
            output = self.fc(output.squeeze(1))
            _, predicted = output.max(1)
            
            # Get the word
            word = vocab.idx2word.get(predicted.item(), '<unk>')
            
            # Stop if we hit the end token
            if word == '<end>':
                break
            
            # Add to caption
            if word != '<start>':
                captions.append(word)
            
            # Prepare next input
            inputs = self.embedding(predicted).unsqueeze(1)
        
        return captions

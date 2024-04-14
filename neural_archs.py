import torch

class DAN(torch.nn.Module):
    def __init__(self, vocab_size, embedding_size, glove_embedding=None):
        super().__init__()
        if glove_embedding is not None:
            self.embedding = glove_embedding
            embedding_size = glove_embedding.embedding_dim
        else:
            self.embedding = torch.nn.Parameter(torch.randn(vocab_size, embedding_size))
            self.embedding.requires_grad_(True)
        
        self.fc = torch.nn.Linear(embedding_size, 1) 
    
    def average(self, x):

        weight = self.embedding.weight if isinstance(self.embedding, torch.nn.Embedding) else self.embedding

        # Perform embedding lookup
        embedded = torch.nn.functional.embedding(x, weight)
        
        # Compute the mean along the sequence dimension
        emb_sum = torch.sum(embedded, dim=1)  # Sum along the sequence dimension
        lengths = (x != 0).sum(dim=1, keepdim=True).float()  # Calculate the lengths of non-zero elements
        emb_mean = emb_sum / lengths  # Calculate mean by dividing by sequence lengths
        
        return emb_mean
        
    def forward(self, x):
        review_averaged = self.average(x)
        out = self.fc(review_averaged)
        return out
    


class RNN(torch.nn.Module):
    def __init__(self, input_size,embedding_dim, hidden_dim, output_size, bidirectional=False, glove_embedding=None):
        super(RNN, self).__init__()
        self.hidden_dim  = hidden_dim
        self.bidirectional = bidirectional
        
        if glove_embedding is not None:
            self.embedding = glove_embedding
            embedding_dim = glove_embedding.embedding_dim
            
        else:
            self.embedding = torch.nn.Embedding(input_size, embedding_dim)

        self.rnn = torch.nn.RNN(embedding_dim, hidden_dim, num_layers=1, bidirectional=bidirectional, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim * (2 if bidirectional else 1), output_size)

    def forward(self, x):
        x_embedded = self.embedding(x)
        output, _ = self.rnn(x_embedded)
        if self.bidirectional:
            x_out = torch.sigmoid(self.fc(torch.cat((output[-2, :, :self.hidden_dim], output[-1, :, self.hidden_dim:]), 1)))
        else:
            x_out = torch.sigmoid(self.fc(output[:, -1, :]))

        return x_out
    
    

class LSTM(torch.nn.Module):
    def __init__(self, input_size,embedding_dim, hidden_dim, output_size, bidirectional=False, glove_embedding=None):
        super(LSTM, self).__init__()
        self.hidden_dim  = hidden_dim
        self.bidirectional = bidirectional

        if glove_embedding is not None:
            embedding_dim = glove_embedding.embedding_dim
            self.embedding = glove_embedding
        else:
            self.embedding = torch.nn.Embedding(input_size, embedding_dim)

        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim,num_layers=1, bidirectional=bidirectional, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim * (2 if bidirectional else 1), output_size)

    def forward(self, x):
        x_embedded = self.embedding(x)
        output, _ = self.lstm(x_embedded)
        if self.bidirectional:
            x_out = torch.sigmoid(self.fc(torch.cat((output[-1, :, :self.hidden_dim], output[0, :, self.hidden_dim:]), 1)))
        else:
            x_out = torch.sigmoid(self.fc(output[:, -1, :]))

        return x_out

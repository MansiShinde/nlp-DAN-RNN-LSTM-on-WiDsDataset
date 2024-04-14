import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import random
random.seed(577)

import numpy as np
np.random.seed(577)

import torch
torch.set_default_tensor_type(torch.FloatTensor)
torch.use_deterministic_algorithms(True)
torch.manual_seed(577)
torch_device = torch.device("cpu")

'''
NOTE: Do not change any of the statements above regarding random/numpy/pytorch.
You can import other built-in libraries (e.g. collections) or pre-specified external libraries
such as pandas, nltk and gensim below. 
Also, if you'd like to declare some helper functions, please do so in utils.py and
change the last import statement below.
'''

import gensim.downloader as api
from torch.utils.data import DataLoader
from neural_archs import DAN, RNN, LSTM
from utils import WiCDataset

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    count = 0
    for batch in train_loader:
        labels = batch['labels'].unsqueeze(1).to(device)
        example = batch['example'].to(device)

        optimizer.zero_grad()
        outputs = model(example)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Get the model's prediction 
        pred = (torch.sigmoid(outputs) > 0.5).float()

        # Update the running statistics
        running_acc += torch.sum((pred == labels).float()).item()
        running_loss += loss.item()
        count += example.size(0)

    return running_acc, running_loss, count


def evaluate(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    count = 0
    with torch.no_grad():
        for batch in data_loader:
            example = batch['example'].to(device)
            labels = batch['labels'].unsqueeze(1).to(device)

            outputs = model(example)
            loss = criterion(outputs, labels)

            # Get the model's prediction 
            pred = (torch.sigmoid(outputs) > 0.5).float()

            # Update the running statistics
            running_acc += torch.sum((pred == labels).float()).item()
            running_loss += loss.item()
            count += example.size(0)

    return running_acc, running_loss, count


def main(args):
    # import logging
    # logging.basicConfig(
    #     filename=f'logs/train_{args.neural_arch}.log',
    #     level=logging.INFO,
    #     format='%(asctime)s - %(levelname)s - %(message)s',
    #     datefmt='%Y-%m-%d %H:%M:%S'  # Customize the timestamp format
    # )

    if args.init_word_embs == "glove":
        print("Loading glove embeddings from Gensim ...")
        glove_embs = api.load("glove-wiki-gigaword-50")
        print(f"Building embeddings for training ...")
        glove_embedding = torch.nn.Embedding.from_pretrained(torch.tensor(glove_embs.vectors, dtype=torch.float32), freeze=True)
    
    # Set parameters
    batch_size = 32
    vocab_size = 10000  # Example vocabulary size
    embedding_dim = 300  # Example embedding dimension for GloVe
    hidden_dim = 128
    num_classes = 1
    num_epochs = 100
    learning_rate = 0.001
    

    # Initialize model based on selected neural architecture
    if args.neural_arch == "dan":
        model = DAN(vocab_size, embedding_dim, glove_embedding=glove_embedding).to(torch_device)
    elif args.neural_arch == "rnn":
        model = RNN(vocab_size, embedding_dim, hidden_dim, num_classes, args.rnn_bidirect, glove_embedding=glove_embedding).to(torch_device)
    elif args.neural_arch == "lstm":
        model = LSTM(vocab_size, embedding_dim, hidden_dim, num_classes, args.rnn_bidirect, glove_embedding=glove_embedding).to(torch_device)
    else:
        raise ValueError(f"Invalid neural_arch specified: {args.neural_arch}")


    # Loss and optimizer
    print(f"Initializing loss and optimizer ...")
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    print(f"Initializing training dataset ...")
    train_dataset = WiCDataset(
                        data_file='WiC_dataset/train/train.data.txt',
                        gold_file='WiC_dataset/train/train.gold.txt'
                    )
    print(f"Initializing training dataloader dataset ...")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
    for epoch in range(num_epochs):
        running_acc, running_loss, count = train(model, train_loader, criterion, optimizer, torch_device)

        print('Epoch [{} / {}] Average Training Accuracy: {:4f}'.format(epoch + 1, num_epochs, running_acc / count))
        print('Epoch [{} / {}] Average Training loss: {:4f}'.format(epoch + 1, num_epochs, running_loss / len(train_loader)))

        #print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}")

    # Testing loop
    print(f"Initializing testing dataset ...")
    test_dataset = WiCDataset(
                        data_file='WiC_dataset/test/test.data.txt',
                        gold_file='WiC_dataset/test/test.gold.txt'
                    )
    print(f"Initializing testing dataloader dataset ...")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=test_dataset.collate_fn)
    running_acc, running_loss, count = evaluate(model, test_loader, criterion, torch_device)
    print('Epoch [{} / {}] Average Training Accuracy: {:4f}'.format(epoch + 1, num_epochs, running_acc / count))
    print('Epoch [{} / {}] Average Training loss: {:4f}'.format(epoch + 1, num_epochs, running_loss / len(train_loader)))


    # Write predictions to file
    print(f"Writing predictions ...")
    with open('test.pred.txt', 'w') as f:
        for batch in test_loader:
            example = batch['example'].to(torch_device)
            outputs = model(example)
            predictions = torch.sigmoid(outputs).detach().cpu().numpy()

            print(predictions)
            for pred in predictions:
                f.write('T\n' if pred >= 0.5 else 'F\n')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--neural_arch', choices=['dan', 'rnn', 'lstm'], default='dan', type=str)
    parser.add_argument('--rnn_bidirect', default=False, action='store_true')
    parser.add_argument('--init_word_embs', choices=['scratch', 'glove'], default='glove', type=str)

    args = parser.parse_args()
    main(args)

    
from torch.utils.data import Dataset
import pandas as pd
import torch
from collections import defaultdict
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn

class WiCDataset(Dataset):
    def __init__(self, data_file, gold_file):
        self.data = pd.read_csv(data_file, sep='\t', header=None, names=['target_word', 'pos', 'indices', 'example1', 'example2'])
        self.gold_labels = pd.read_csv(gold_file, header=None, names=['label'])

        # Initialize vocabulary and POS mappings
        self.vocab_mapping = defaultdict(lambda: len(self.vocab_mapping))  # Assigns unique index to each new word
        self.pos_mapping = {'N': 0, 'V': 1, 'ADJ': 2, 'ADV': 3, 'P': 4}  # Assigns indices to specific POS tags

        # Build vocabulary and POS mappings
        self.build_vocab_and_pos_mappings()

    def build_vocab_and_pos_mappings(self):
        # Iterate over each row in the dataset to update vocabulary and POS mappings
        for idx, row in self.data.iterrows():
            # Tokenize example1 and example2 to update vocabulary
            tokens1 = word_tokenize(row['example1'])
            tokens2 = word_tokenize(row['example2'])
            
            for token in tokens1 + tokens2:
                self.vocab_mapping[token]  # Assigns a unique index to each token if not present in the mapping

            # Update POS mapping
            pos_tag = row['pos']
            if pos_tag not in self.pos_mapping:
                self.pos_mapping[pos_tag] = len(self.pos_mapping)  # Assigns a new index to unseen POS tags

    def compute_semantic_similarity(self, target_word, context):
        # Initialize similarity score
        max_similarity = 0.0
        
        # Iterate over each word in the context
        for word in word_tokenize(context):
            # Get synsets (sets of synonyms) for both target word and context word
            synsets_target = wn.synsets(target_word)
            synsets_context = wn.synsets(word)
            
            # Calculate similarity between all pairs of synsets and take the maximum
            for synset_target in synsets_target:
                for synset_context in synsets_context:
                    similarity = synset_target.path_similarity(synset_context)
                    if similarity is not None and similarity > max_similarity:
                        max_similarity = similarity
        
        return max_similarity if max_similarity else 0.0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_row = self.data.iloc[idx]
        target_word = data_row['target_word']
        pos = self.pos_mapping[data_row['pos']]
        index1, index2 = map(int, data_row['indices'].split('-'))
        example1 = data_row['example1']
        example2 = data_row['example2']
        label = 1 if self.gold_labels.iloc[idx]['label'] == 'T' else 0

        # Compute semantic similarity between target word and examples
        similarity_example1 = self.compute_semantic_similarity(target_word, example1)
        similarity_example2 = self.compute_semantic_similarity(target_word, example2)


        return {
            'target_word': target_word,
            'pos': pos,
            'index1': index1,
            'index2': index2,
            'example1': example1,
            'example2': example2,
            'similarity_example1': similarity_example1,  # New feature
            'similarity_example2': similarity_example2,  # New feature
            'label': label
        }

    def collate_fn(self, batch):
        examples1 = [torch.tensor([self.vocab_mapping[word] for word in item['example1'].split()], dtype=torch.long) for item in batch]
        examples2 = [torch.tensor([self.vocab_mapping[word] for word in item['example2'].split()], dtype=torch.long) for item in batch]
        pos = [item['pos'] for item in batch]
        labels = [torch.tensor(item['label'], dtype=torch.float) for item in batch]

        # Create a single array with example1, example2, pos
        examples = [torch.cat((example1, example2, torch.tensor([pos_])), dim=0) for example1, example2, pos_ in zip(examples1, examples2, pos)]
        # Group them into batch
        examples = torch.nn.utils.rnn.pad_sequence(examples, batch_first=True)

        return {
            'example': examples,  # Store as 'example'
            'labels': torch.stack(labels) if labels else torch.tensor([])
        }

        # return {
        #     'example1': torch.nn.utils.rnn.pad_sequence(examples1, batch_first=True),
        #     'example2': torch.nn.utils.rnn.pad_sequence(examples2, batch_first=True),
        #     'pos': torch.tensor(pos, dtype=torch.long),
        #     'labels': torch.stack(labels) if labels else torch.tensor([])
        # }

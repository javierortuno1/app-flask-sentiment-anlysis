from typing import Optional  # Add this at the top with other imports
import os
import torch
from torch.utils.data import Dataset
from torchtext.data import get_tokenizer, to_map_style_dataset
from torchtext.data.utils import ngrams_iterator
from torchtext.vocab import build_vocab_from_iterator
import pandas as pd

class LocalYelpDataset(Dataset):
    """Custom Dataset for loading local Yelp Review files"""
    def __init__(self, filename):
        self.data = pd.read_csv(filename, names=['label', 'text'])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data.iloc[idx]['label'], self.data.iloc[idx]['text']


class YelpReviewPolarityDatasetLoader(object):
    """
    Yelp Review Polarity dataset loader helper class.
    """
    def __init__(self, ngrams: int = 2, batch_size: int = 16, data_dir: str = "data",
                 device: Optional[torch.device] = None):
        # Dataset loader hyper-parameters
        self.NGRAMS = ngrams
        self.BATCH_SIZE = batch_size
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        # Define internal class variables
        self._tokenizer = get_tokenizer("basic_english")
        self._data_dir = data_dir
        
        # Create data dir if it doesn't exist
        if not os.path.isdir(self._data_dir):
            os.mkdir(self._data_dir)
        
        # Load local datasets
        train_iter = LocalYelpDataset(os.path.join(self._data_dir, 'train.csv'))
        
        # Build vocabulary from iterator
        self.vocab = build_vocab_from_iterator(self._yield_tokens(train_iter), specials=["<unk>"])
        self.vocab.set_default_index(self.vocab["<unk>"])

    def _yield_tokens(self, data_iter):
        for _, text in data_iter:
            yield ngrams_iterator(self._tokenizer(text), self.NGRAMS)

    def get_num_classes(self):
        train_iter = LocalYelpDataset(os.path.join(self._data_dir, 'train.csv'))
        return len(set([label for (label, _) in train_iter]))

    def get_vocab_size(self):
        return len(self.vocab)

    def _get_dataset(self, split: str):
        filename = os.path.join(self._data_dir, f'{split}.csv')
        dataset = LocalYelpDataset(filename)
        return dataset

    def get_train_val_dataset(self):
        return self._get_dataset(split="train")

    def get_test_dataset(self):
        return self._get_dataset(split="test")

    def _text_pipeline(self, x):
        return self.vocab(list(ngrams_iterator(self._tokenizer(x), self.NGRAMS)))

    @staticmethod
    def _label_pipeline(x):
        return int(x) - 1

    def generate_batch(self, batch):
        label_list, text_list, offsets = [], [], [0]
        for label, text in batch:
            label_list.append(self._label_pipeline(label))
            processed_text = torch.tensor(self._text_pipeline(text), dtype=torch.int64)
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)
        return text_list.to(self.device), offsets.to(self.device), label_list.to(self.device)
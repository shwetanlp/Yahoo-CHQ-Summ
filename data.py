# Importing stock libraries
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler


from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'


class CustomDataset(Dataset):
    def __init__(self, dataset , tokenizer, source_len, summ_len):
        self.tokenizer = tokenizer
        self.source_len = source_len
        self.summ_len = summ_len
        x_seq, y_seq = get_seq(dataset)
        self.ctext = x_seq
        self.text = y_seq

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        ctext = self.ctext[index]
        text = self.text[index]
        source = self.tokenizer.batch_encode_plus([ctext], max_length= self.source_len, pad_to_max_length=True,return_tensors='pt')
        target = self.tokenizer.batch_encode_plus([text], max_length= self.summ_len, pad_to_max_length=True,return_tensors='pt')

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long),
            'source_mask': source_mask.to(dtype=torch.long),
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long),
            'target_mask': target_mask.to(dtype=torch.long)
        }


def get_seq(data):
    x_seq = []
    y_seq = []
    for d in data:
        x_seq.append(d["x"])
        y_seq.append(d["y"])
    return x_seq, y_seq

def read_langs(file_name, words=None, mode='train', max_src=None, max_tgt=None):
    data = []
    with open(file_name[0], "r") as f1, open(file_name[1], "r") as f2:
        for src_line, tgt_line in zip(f1.readlines(), f2.readlines()):  ##5040
            src = src_line.strip()
            tgt = tgt_line.strip()
            d = {}
            d["x"] = src
            d["y"] = tgt

            if max_src is not None:
                d["x"] = ' '.join(d["x"].strip().split()[:max_src])

            if max_tgt is not None:
                d["y"] = ' '.join(d["y"].strip().split()[:max_tgt])

            d["x_len"] = len(d["x"].strip().split())
            d["y_len"] = len(d["y"].strip().split())
            data.append(d)
    max_src = max([d["x_len"] for d in data])
    max_tgt = max([d["y_len"] for d in data])
    print(f"Total data size: {len(data)}")
    return data, max_src, max_tgt



from jsonargparse import ArgumentParser
from torch.utils.data import Dataset, Sampler, DataLoader
import torch
import pandas as pd

def setup_seed(seed):
    import numpy as np
    import random
    from torch.backends import cudnn
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def init_args():
    parser = ArgumentParser(default_config_files=['config.yaml'])
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--datasize', type=str, default='demo',
                        help='size: {demo, full}')
    parser.add_argument('--batchsize', type=int, default=1024)
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device: {cuda, cpu}')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight for L2 loss')    
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='dropout probability')
    parser.add_argument('--hidden_size', type=list, default=[32, 16],
                        help='hidden_size')
    parser.add_argument('--features')
    parser.add_argument('--embedding')
    args = parser.parse_args()
    args.use_feats = {vs: k for k, v in args.features.items() for vs in v}
    return args


class MCC_Dataset(Dataset):
    def __init__(self, file, args):
        self.args = args
        self.file = file
        self.cols = list(args.use_feats.keys())
        self.convert = {key: lambda x: list(map(int, x.split(',')))
                        for key in args.use_feats.keys() if 'seq' in key}

    def reset(self):
        self.data = pd.read_csv(self.file, usecols=self.cols,
                                converters=self.convert, iterator=True)

    def __getitem__(self, batchsize):
        x = self.data.get_chunk(batchsize)
        x = x.to_dict(orient='list')
        for k in self.args.use_feats.keys():
            x[k] = torch.tensor(x[k])
        return x

    def __len__(self):
        if self.args.datasize == 'demo':
            return 10000
        elif 'train' in self.file:
            return 9526571
        elif 'test' in self.file:
            return 3555419


def data_loader(dataset, batchsize=None):
    sampler = CSV_Sampler(dataset)
    batch_sampler = CSV_Batch_Sampler(sampler, batchsize)
    return DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=lambda x: x[0])

class CSV_Sampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return self.data_source.data

    def __len__(self):
        return len(self.data_source)


class CSV_Batch_Sampler(Sampler):
    def __init__(self, csv_sampler, batchsize):
        self.sampler = csv_sampler
        self.batch_size = batchsize

    def __iter__(self):
        return self

    def __next__(self):
        return [self.batch_size]

    def __len__(self):
        return (len(self.sampler) + self.batch_size - 1) // self.batch_size




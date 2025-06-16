import os
import os.path as osp
import shutil
import pickle

import torch
from tqdm import tqdm
from torch_geometric.data import (InMemoryDataset, Data, download_url,
                                  extract_zip)

class EVENTLOG(InMemoryDataset):

    def __init__(self, root, split='train', transform=None, pre_transform=None,
                 pre_filter=None):
        self.name = "EVENTBPIC15M1"
        assert split in ['train', 'val', 'test']
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return ['train.pickle', 'val.pickle', 'test.pickle']

    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def process(self):
        for split in ['train', 'val', 'test']:
            with open(osp.join(self.raw_dir, f'{split}.pickle'), 'rb') as f:
                graphs = pickle.load(f)

            indices = range(len(graphs))

            pbar = tqdm(total=len(indices))
            pbar.set_description(f'Processing {split} dataset')

            data_list = []
            for idx in indices:
                graph = graphs[idx]
                x = graph.x
                edge_attr = graph.edge_attr
                edge_index = graph.edge_index
                y = graph.y
                cid = graph.cid
                pl = graph.pl

                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, cid=cid, pl=pl)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                pbar.update(1)

            pbar.close()
            torch.save(self.collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))

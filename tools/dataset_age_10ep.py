import torch
import torch.utils.data as data
import os
import pickle
from os.path import join
from sklearn.model_selection import train_test_split
import numpy as np
from glob import glob

class XY_cohort_seq(data.Dataset):
    def __init__(self, cfg, tvt = 'train', seq_len = 10, epoch_num = 1260):
    ###
    # tvt: 'train', 'valid', 'test', 'all'
    ###
        super(XY_cohort_seq, self).__init__()
        self.seq_len = seq_len
        self.tvt = tvt
        self.frame_len = cfg.freq * 30
        self.channel_num = cfg.SWIN.IN_CHANS
        self.epoch_num = float(epoch_num)
        dataset = cfg.dataset
        if type(dataset) is str:
            savName = dataset
            dataset = [dataset]
        else:
            savName = 'Custom{:02d}'.format(len(dataset))
        redir_cache, redir_root = cfg.redir_cache, cfg.redir_root
        if not redir_cache:
            cache_path = join('experiments/{:}/prepared_data/{:}_{:}_cache.pkl'.format(cfg.experiment, tvt, savName))
        else:
            cache_path = join(redir_cache, '{:}_{:}_cache.pkl'.format(tvt, savName))
        #if cache
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                cache = pickle.load(f)
            if not redir_root:
                self.root = cache['root']
            else:
                self.root = redir_root
            self.items_psg = cache['items_psg']
            self.items_pkl = cache['items_pkl']
            self.dict_hyp = cache['hyp']
            self.dict_age = cache['age']
            self.len = len(self.items_psg)
            return
        #else
        
        # subject selector
        if not redir_root:
            self.root = [r'G:\data\filtered_data_128\subjects', r'E:\data\filtered_data_128\subjects']
        else:
            self.root = redir_root
        self.subjIdx = {
            'SHHS1': (0, 5666),
            'SHHS2': (5667, 8287),
            'CFS': (9252, 9973),
            'MROS1': (9974, 12851),
            'MROS2': (12852, 13859),
            'MESA': (13860, 15893),
            'HPAP1':(15894, 16083),
            'ABC1':(16139, 16187),
            'ABC2':(16188, 16229),
            'ABC3':(16230, 16269),
            'NCHSDB':(16270, 17251),
            'WSC':(19903, 22436),
            }

        # split
        index_psgs = []
        if tvt == 'all':
            for d in dataset:
                for idx in range(self.subjIdx[d][0], self.subjIdx[d][1] + 1):
                    root = self.root[0] if idx<15000 else self.root[1]
                    index_psg = join(root, '{:06d}'.format(idx))
                    assert os.path.exists(index_psg)
                    index_psgs.append(idx)
        else:
            from tools.data_tools import Split        
            train_idx, valid_idx, test_idx = Split().split_dataset(dataset)
            psg_idx = train_idx if tvt == 'train' else valid_idx if tvt == 'valid' else test_idx if tvt == 'test' else None
            for idx in psg_idx:
                root = self.root[0] if idx<15000 else self.root[1]
                index_psg = join(root, '{:06d}'.format(idx))
                assert os.path.exists(index_psg)
                index_psgs.append(idx)

        # load age in memory (idx_age)
        with open(r'E:\data\filtered_data_128\subjects.info', 'r') as f:
            lines = f.readlines()[1:]
        self.dict_age = {}
        for line in lines:
            # readline
            idx, dataset, nsrrid, sex, age, length = line.strip().split(', ')
            self.dict_age[int(idx)] = float(age)

        # load other prepared items
        self.items_psg, self.items_pkl, self.dict_hyp = [], [], {}
        for index_psg in index_psgs:
            root = self.root[0] if index_psg<15000 else self.root[1]
            # dict_hyp
            if os.path.exists(join(root, '%06d'%index_psg, 'pred_hypnodensity.pkl')):
                with open(join(root, '%06d'%index_psg, 'pred_hypnodensity.pkl'), 'rb') as f:
                    hypnodensity = pickle.load(f)[:,:5].softmax(1)
                frameNum = len(hypnodensity)
                # set hypnodensity length to 1260
                if len(hypnodensity) <= epoch_num:
                    hyp_scaled = torch.tensor([1., 0., 0., 0., 0.]).repeat(1260, 1)
                    hyp_scaled[:len(hypnodensity)] = hypnodensity
                else:
                    hyp_scaled = hypnodensity[:epoch_num]
            else:
                frameNum = len(glob(join(root, '%06d'%index_psg, 'data/*[0-9].pkl')))
                hyp_scaled = torch.zeros(epoch_num, 5) + .2
            self.dict_hyp[index_psg] = (hyp_scaled)
            for idx in range(0, frameNum - seq_len, seq_len//2):
                self.items_pkl.append(idx)
                self.items_psg.append(index_psg)

        self.len = len(self.items_psg)
        # save cache
        cache = {'root':self.root, 'items_psg': self.items_psg,'items_pkl':self.items_pkl, 
                 'hyp': self.dict_hyp, 'age': self.dict_age}
        with open(cache_path, 'wb') as f:
            pickle.dump(cache, f)

    def __getitem__(self, index):##########################################################
        # with torch.autograd.profiler.profile(enabled=True) as prof:
        index_pkl = self.items_pkl[index]
        index_psg = self.items_psg[index]
        root = self.root[0] if index_psg<15000 else self.root[1] 
        paths = ['{:}\\{:06d}\\data\\{:04d}.pkl'.format(root, index_psg, idx) for idx in range(index_pkl, index_pkl + self.seq_len)]
        # load EEG
        X_EEG = torch.zeros(size = [self.seq_len * self.frame_len, self.channel_num]).float()
        for i in range(self.seq_len):
            with open(paths[i], 'rb') as f_data:
                pkl = pickle.load(f_data)[:,:3]
            X_EEG[i*self.frame_len:(i+1)*self.frame_len,:] = torch.from_numpy(pkl).float()
        X_EEG = torch.clip(X_EEG, -1000, 1000)
        X_EEG = torch.swapaxes(X_EEG, 0, 1) # B, C, L
        
        # load HYP
        X_HYP = self.dict_hyp[index_psg]
        X_HYP = torch.swapaxes(X_HYP, 0, 1) # B, C, L
        location = torch.tensor(index_pkl / self.epoch_num).float()
        age = torch.tensor(self.dict_age[index_psg]).float()

        # print(prof.key_averages().table(sort_by='self_cpu_time_total'))
        return X_EEG, X_HYP, location, age, index_psg

    def __len__(self):
        return self.len


if __name__ == '__main__':
    pass



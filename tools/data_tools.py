from sklearn.model_selection import train_test_split
import torch,os, pickle, threading
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

class MedianAge(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x):
        x = x.softmax(-1)
        stairs = x.unsqueeze(-1).expand(-1,-1,100).reshape(x.size(0), -1)

        # forward
        cumsumed_probs_forward = torch.cumsum(stairs, dim = -1) / 100
        median_age_forward = torch.argmax((cumsumed_probs_forward >= 0.5).int(), dim=1).float() / 10

        # reverse
        cumsumed_probs_backward = torch.cumsum(torch.flip(stairs, dims=[1]), dim=-1) / 100
        median_idx_backward = torch.argmax((cumsumed_probs_backward >= 0.5).int(), dim=1).float()
        median_age_backward = 100 - median_idx_backward / 10

        # average
        median_age = torch.mean(torch.stack([median_age_forward, median_age_backward],0),0)
        return median_age

class _fft_surrogate(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, x, beta):
        f = torch.fft.fft(x)
        n = f.shape[0]
        random_phase = 2j*torch.pi*torch.rand(n//2-1) * beta
        random_phase = torch.cat([torch.Tensor([0]), random_phase, torch.Tensor([0]), -torch.flip(random_phase,[0])])
        f_shifted = f*torch.exp(random_phase)
        shifted = torch.fft.ifft(f_shifted)
        return shifted.float()

class _fft_shift(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def myHilbert(self, x, f):
        n = x.shape[0]
        fa = torch.fft.fftshift(f)
        fa[:n//2] *= 1j
        fa[n//2:] *= -1j
        z = torch.fft.ifftshift(fa)
        z[0] = 0.0
        z[n//2] = 0
        return torch.abs(torch.fft.ifft(z).real)
    def forward(self, x, beta = .1):
        f = torch.fft.fft(x)
        n = f.shape[0]
        xa = x + 1j*self.myHilbert(x, f)
        random_phase = 2j*torch.pi*torch.rand(n//2-1).cpu() * beta
        random_phase = torch.cat([torch.Tensor([0]).cpu(), random_phase, torch.Tensor([0]).cpu(), -torch.flip(random_phase,[0])])
        x_shifted = xa*torch.exp(random_phase)
        x_shifted = x_shifted.real
        return x_shifted.float()

class _gauss_noisy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, x, beta):
        # x shape = [n_sig, length]
        noise = torch.randn_like(x)
        std = beta.unsqueeze(1) * torch.std(x, dim=1, keepdim=True)

        # 给每行的信号添加噪声
        x_noisy = x + std * noise
        return x_noisy

class DataAugment(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()
        self.fft_surrogate = _fft_surrogate()
        self.fft_shift = _fft_shift()
        self.gauss_noise = _gauss_noisy()
        
    def forward(self, X):
        B, C, L = X.shape
        rand_table = torch.rand([B, 16]).cuda()

        # # Spacial
        # # O2 -> F4 Channel Overlay: 10%
        # idx = rand_table[:,0] < 0.1
        # X[idx,0,:] += 3 * rand_table[idx, 0].unsqueeze(1) * X[idx,-2,:]   # beta = 0~30%

        # EEG Reverse: 10%
        X[rand_table[:,1] < 0.1,:3,:] *= -1
        # # EOG Reverse: 10%
        # X[rand_table[:,2] < 0.1,3:5,:] *= -1
        # # EMG Reverse: 10%
        # X[rand_table[:,3] < 0.1,-1,:] *= -1

        # channel missing: 10%
        # idx = rand_table[:,4:10] < 0.1
        idx = rand_table[:,4:7] < 0.1
        idx[:,1] = False
        X[idx,:] = 0

        # Temporal
        # Gauss Noisy: 10%
        for i in range(3):
            idx = rand_table[:, i+10] < 0.1
            beta = rand_table[idx, i+10] * 2    # beta = 0~20% std
            X[idx,i,:] = self.gauss_noise(X[idx,i,:], beta)  

        return X

class FeatAugment(nn.Module):
    def __init__(self, scale_range=(0.9, 1.1), shift_range=(-0.1, 0.1), noise_std=0.1):
        super().__init__()
        self.scale_range = scale_range
        # self.shift_range = shift_range
        # self.noise_std = noise_std

    def forward(self, feat):
        B, C = feat.size()
        # Scaling
        scales = torch.FloatTensor(B, C).uniform_(*self.scale_range)
        feat = feat * scales
        # # Shifting
        # feat_std = torch.std(feat, dim=0)
        # shifts = torch.FloatTensor(B, C).uniform_(*self.shift_range)
        # feat = feat + shifts * feat_std
        # # Gaussian noise
        # noise = torch.randn(B, C) * self.noise_std
        # feat = feat + noise * feat_std
        return feat

class Split():
    def __init__(self) -> None:
        self.set_idx = {
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
        self.cohort_subj_visit = self._subjInCohorts()

        # presplit cohorts
        self.cohort_tvt = {}
        for cohort in self.cohort_subj_visit:
            tvt = self._presplit_cohort(cohort)
            self.cohort_tvt[cohort] = tvt

    # output: {cohort:{nid1:[idx1, idx2, ...], ...]}}
    def _subjInCohorts(self):
        def nsrr2subj(name:str):
            sp = name.split('-')
            return sp[-2] if 'nsrr' in sp[-1] else sp[-1]
        
        cohort_subj = {}   # format: {cohort:{nid1:[idx1, idx2, ...], ...]}}
        with open(r'E:\data\filtered_data_128\subjects.info', 'r') as f:
            lines = f.readlines()[1:]
        for line in lines:
            # readline
            idx, dataset, nsrrid, sex, age, length = line.strip().split(', ')
            # skip no age recorded  ###################
            if float(age) < 0:
                continue
            # init cohort, nid
            cohort = dataset[:-1] if dataset[-1] == '1' or dataset[-1] == '2' or dataset[-1]=='3' else dataset  # set name to cohort name
            if cohort not in cohort_subj:
                cohort_subj[cohort] = {}
            subjid = nsrr2subj(nsrrid)
            if subjid not in cohort_subj[cohort]:
                cohort_subj[cohort][subjid] = []
            # store 
            cohort_subj[cohort][subjid].append(int(idx))

        # print
        print('\ndataset\tsubject num')
        for cohort in cohort_subj:
            print('%s\t%d'%(cohort, len(cohort_subj[cohort])))
        print('\ndataset\trecord num')  
        for dataset in self.set_idx:
            print('%s\t%d'%(dataset, self.set_idx[dataset][1] - self.set_idx[dataset][0] + 1))
        return cohort_subj

    # split one cohort 
    # (valid: 10% subj, up to 50)
    # (test: 15% subj, up to 100)
    def _presplit_cohort(self, cohort:str):
        n_subj = len(self.cohort_subj_visit[cohort])
        n_test = min(100, round(n_subj * .15))
        n_valid = min(50, round(n_subj * .10))
        train_subj, test_subj = train_test_split(list(self.cohort_subj_visit[cohort]), 
                test_size = n_test, random_state = 0)
        train_subj, valid_subj = train_test_split(train_subj, 
                test_size = n_valid, random_state = 0)
        return train_subj, valid_subj, test_subj

    # BUG: cannot input single SHHS1/SHHS2, MASS1/MASS2 or ABC1/2/3
    def split_dataset(self, datasets:list):
        cohorts = []
        for dataset in datasets:
            cohort = dataset[:-1] if dataset[-1].isdigit() else dataset  # set name to cohort name
            cohorts.append(cohort)
        train_idx, valid_idx, test_idx = [], [], []
        for c in set(cohorts):
            tvt_subj = self.cohort_tvt[c]
            # train..
            for s in tvt_subj[0]:
                train_idx += self.cohort_subj_visit[c][s]
            # valid..
            for s in tvt_subj[1]:
                valid_idx += self.cohort_subj_visit[c][s]
            # test..
            for s in tvt_subj[2]:
                test_idx += self.cohort_subj_visit[c][s]
            print('%s,\ttrain:%d, valid:%d, test%d'%(c, len(tvt_subj[0]), len(tvt_subj[1]), len(tvt_subj[2])))
        print('\nSelected:%d, %d, %d'%(len(train_idx), len(valid_idx), len(test_idx)))
        return (sorted(train_idx), sorted(valid_idx), sorted(test_idx))

if __name__ == '__main__':
    pass
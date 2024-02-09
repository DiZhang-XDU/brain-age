import os,time,pickle
import numpy as np
import torch
from timm import scheduler as schedulers    # do not delete!

from os.path import join
from tqdm import tqdm
from torch.nn import DataParallel as DP
from tools import *
from sleep_models.heads import HeadPost

import warnings
warnings.filterwarnings("ignore")##################################
import multiprocessing
multiprocessing.set_start_method('spawn', True)

class FeatureSet(torch.utils.data.Dataset):
    def __init__(self, psg_feat = None, train = True):
        super().__init__()
        self.train = train
        # load age in memory 
        psg_age = {}
        with open(r'E:\data\filtered_data_128\subjects.info', 'r') as f:
            lines = f.readlines()[1:]
        for line in lines:
            # readline
            idx, dataset, nsrrid, sex, age, length = line.strip().split(', ')
            psg_age[int(idx)] = float(age)

        self.psg_idx, self.psg_feat, self.psg_age = [], [], []
        for psg in psg_feat:
            self.psg_idx.append(psg)
            self.psg_feat.append(psg_feat[psg])
            self.psg_age.append(psg_age[psg])
        
    def __getitem__(self, index, psg_len = 1260//5):
        feat = self.psg_feat[index]
        if len(feat) <= psg_len:
            x = torch.cat([feat, torch.zeros([psg_len - len(feat), feat.size(1)])]).float()
        elif self.train:
            start = int((len(feat) - psg_len) * torch.rand(1))
            x = feat[start:start+psg_len, :]
        else:
            x = feat[:psg_len]
        x = x.swapaxes(0,1) # BLC -> BCL
        y = self.psg_age[index]
        return x, y, self.psg_idx[index]
    
    def __len__(self):
        return len(self.psg_feat)

def trainval(cfg):
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.CUDA_VISIBLE_DEVICES
    age = MedianAge().cuda()
    age.eval()
    
    # feat loader
    deepFeatPath = r'experiments\%s\prepared_data\trainval_feature.pkl'%cfg.experiment
    if not os.path.exists(deepFeatPath):
        print('Calculating Deep Feature...')
        model_feat = torch.load(cfg.best_ckp)['net'].cuda()
        model_feat.eval()

        myDataset = datasets(cfg)
        trainSet = myDataset(cfg, tvt = 'train')
        validSet = myDataset(cfg, tvt = 'valid')

        loader, psg_feat = {}, {}
        loader['train'] = torch.utils.data.DataLoader(trainSet, batch_size = cfg.BATCH_SIZE,
                                        shuffle = False, num_workers = cfg.eval_thread, drop_last = False, pin_memory = False)
        loader['valid'] = torch.utils.data.DataLoader(validSet, batch_size = cfg.BATCH_SIZE,
                                        shuffle = False, num_workers = cfg.eval_thread, drop_last = False, pin_memory = False)

        with torch.no_grad():
            for tvt in loader:
                psg_feat[tvt] = {}
                psg_pred, psg_target = {}, {}
                tq = tqdm(loader[tvt], desc= tvt, ncols=80, ascii=True)
                for i, (x_eeg, x_hyp, x_loc, target, psg) in enumerate(tq):
                    with torch.no_grad(): 
                        p, f = model_feat(x_eeg.cuda(),x_hyp.cuda(),x_loc.cuda())
                        pred = age(p)

                    pred = pred.cpu()
                    feature = f.cpu().detach().clone().view([f.size(0),-1])
                    for j in range(len(psg)):
                        idx_psg = int(psg[j])
                        if idx_psg not in psg_feat[tvt]:
                            psg_pred[idx_psg] = torch.empty([0]).int()
                            psg_target[idx_psg] = int(target[j])
                            psg_feat[tvt][idx_psg] = torch.empty([0, f.size(1)])
                        psg_pred[idx_psg] = torch.cat([psg_pred[idx_psg], pred[j:j+1].int()])
                        psg_feat[tvt][idx_psg] = torch.cat([psg_feat[tvt][idx_psg], feature[j:j+1]])
                # performance
                y_true, y_pred, idx = [], [], []
                for s in psg_target:
                    idx += [[s]] * len(psg_pred[s])
                    y_true += [psg_target[s]] * len(psg_pred[s])
                    y_pred += psg_pred[s]
                idx = np.array(idx)
                y_true = np.array(y_true)
                y_pred = np.array(y_pred)
                report = 'Task:%s\t%s\n'%(cfg.experiment,tvt)
                report += 'MAE=%.2f, p=%.2f\n'%(
                    np.abs(y_true - y_pred).mean(),
                    np.corrcoef(y_true, y_pred)[0,1])
                if not os.path.exists(r'experiments\%s\results'%cfg.experiment):
                    os.mkdir(r'experiments\%s\results'%cfg.experiment)
                np.savetxt('experiments/%s/results/%s.csv'%(cfg.experiment, tvt),
                        np.stack([idx.reshape(-1), y_true, y_pred]).swapaxes(0,1),fmt='%.1f',delimiter=',')
        with open(deepFeatPath,'wb') as f:
            pickle.dump(psg_feat, f)
    else:
        with open(deepFeatPath,'rb') as f:
            psg_feat = pickle.load(f)
    
    # prepare
    EPOCH_MAX = 50
    BATCH_SIZE = 128
    IN_CHANS = 48+10 if 'merge' in cfg.experiment else cfg.SWIN.EMBED_DIM
    model = HeadPost(cfg, in_chans=IN_CHANS).cuda()

    trainSet = FeatureSet(psg_feat['train'])
    validSet = FeatureSet(psg_feat['valid'])
    trainLoader = torch.utils.data.DataLoader(trainSet, batch_size = BATCH_SIZE, persistent_workers=True,
                shuffle = True, num_workers = 1, drop_last = False, pin_memory = False)
    validLoader = torch.utils.data.DataLoader(validSet, batch_size = BATCH_SIZE,
                shuffle = False, num_workers = 0, drop_last = False, pin_memory = False)
    criterion = criterions(cfg).cuda()
    optim = torch.optim.AdamW(model.parameters(), eps = 1e-8, 
            lr = cfg.lr, betas = [0.9, 0.999], weight_decay = cfg.weight_decay)
    scheduler = schedulers.cosine_lr.CosineLRScheduler(
            optim,  t_initial = len(trainLoader) * EPOCH_MAX, 
            lr_min = 1e-6,  warmup_lr_init = 1e-6,  warmup_t = 0.1 * len(trainLoader), 
            cycle_limit = 1,  t_in_epochs=False, )
    bestMAE = 9999.
    for epoch in range(EPOCH_MAX):
        # train
        torch.cuda.empty_cache()
        model.train()
        tq = tqdm(trainLoader, desc= 'train', ncols=80, ascii=True)
        for i, (feat, y, psg) in enumerate(tq):
            p = model(feat.cuda())
            loss = criterion(p, y.cuda())
            with torch.no_grad():
                pred = age(p)
            optim.zero_grad()
            loss.backward()
            optim.step()
            pred = pred.cpu()
            tq.set_postfix({'Loss':'{:.4f}'.format(loss.item()), 
                            'MAE:':'{:.4f}'.format(torch.abs(pred - y).mean().item())})
        # eval
        valid_pred, valid_y = torch.zeros(len(validSet)) - 1, torch.zeros(len(validSet)) - 1
        torch.cuda.empty_cache()
        model.eval()
        tq = tqdm(validLoader, desc='valid', ncols=80, ascii=True)
        for i, (feat, y, psg) in enumerate(tq):
            with torch.no_grad():
                p = model(feat.cuda())
                pred = age(p)
            pred = pred.cpu()
            valid_pred[i*BATCH_SIZE:i*BATCH_SIZE+BATCH_SIZE] = pred
            valid_y[i*BATCH_SIZE:i*BATCH_SIZE+BATCH_SIZE] = y
            tq.set_postfix({'Loss':'{:.4f}'.format(loss.item()), 
                            'MAE:':'{:.4f}'.format(torch.abs(pred - y).mean().item())})
        # epoch end
        assert -1 not in valid_pred
        scheduler.step_update((epoch+1) * len(trainLoader))
        mae = torch.abs(valid_pred - valid_y).mean().item()
        print('epoch %d, MAE = %.2f'%(epoch, mae))
        with open(r'experiments\%s\log_post.txt'%cfg.experiment, 'a') as f:
            f.write('epoch %d: Valid MAE=%.2f\n'%(epoch, mae))
        if mae < bestMAE:
            torch.save({'net': model}, r'experiments\%s\weights\post_checkpoint'%cfg.experiment)

if __name__ == '__main__':
    yaml_path = r'experiments\swin_merge\config.yaml'

    yh = YamlHandler(yaml_path)
    cfg = yh.read_yaml()
    trainval(cfg,)
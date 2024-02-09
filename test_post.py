import warnings
import numpy as np
import torch,os,pickle,time
from os.path import join
from tqdm import tqdm
from torch.nn import DataParallel as DP
from tools import *

np.set_printoptions(suppress=True) 
import warnings
warnings.filterwarnings("ignore")
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

def tester_post(cfg):
    EPOCH_MAX = 50
    BATCH_SIZE = 16
    IN_CHANS = 10 if 'merge' in cfg.experiment else cfg.SWIN.EMBED_DIM

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.CUDA_VISIBLE_DEVICES
    model = torch.load(cfg.post_ckp)['net'].cuda()
    model.eval()
    age = MedianAge().cuda()

    # prepare feature
    deepFeatPath = r'experiments\%s\prepared_data\%s_feature.pkl'%(cfg.experiment, cfg.tvt)
    if not os.path.exists(deepFeatPath) and ('valtest' not in deepFeatPath):
        print('Calculating Deep Feature...')
        model_feat = torch.load(cfg.best_ckp)['net'].cuda()
        model_feat.eval()

        myDataset = datasets(cfg)
        testSet = myDataset(cfg, tvt=cfg.tvt)
        testLoader = torch.utils.data.DataLoader(testSet, batch_size = cfg.BATCH_SIZE*2, 
                        shuffle = False, num_workers = cfg.eval_thread, drop_last = False, pin_memory = True)
        with torch.no_grad():
            psg_features= {}
            tq = tqdm(testLoader, desc= 'test', ncols=80, ascii=True)
            for i, (x_eeg, x_hyp, x_loc, _, psg) in enumerate(tq):
                with torch.no_grad(): 
                    _, f = model_feat(x_eeg.cuda(),x_hyp.cuda(),x_loc.cuda())
                feature = f.cpu().detach().clone().view([f.size(0),-1])

                for j in range(len(psg)):
                    idx_psg = int(psg[j])
                    if idx_psg not in psg_features:
                        psg_features[idx_psg] = torch.empty([0, feature.size(1)])
                    psg_features[idx_psg] = torch.cat(
                        [psg_features[idx_psg], feature[j:j+1]])
        with open(deepFeatPath,'wb') as f:
            pickle.dump(psg_features, f)
    elif 'valtest' in deepFeatPath and os.path.exists(deepFeatPath.replace('valtest','trainval')) and os.path.exists(deepFeatPath.replace('valtest','test')):
        with open(deepFeatPath.replace('valtest','trainval'),'rb') as f:
            psg_features = pickle.load(f)['valid']
        with open(deepFeatPath.replace('valtest','test'),'rb') as f:
            psg_features_1 = pickle.load(f)
        psg_features.update(psg_features_1)
    else:
        with open(deepFeatPath,'rb') as f:
            psg_features = pickle.load(f)

    # eval
    torch.cuda.empty_cache()
    testFeatSet = FeatureSet(psg_features, train=False)
    testFeatLoader = torch.utils.data.DataLoader(testFeatSet, batch_size = BATCH_SIZE,
            shuffle = False, num_workers = 0, drop_last = False, pin_memory = False)
    
    test_pred = torch.zeros(len(testFeatSet)) - 1
    test_y = torch.zeros_like(test_pred) - 1
    test_psg = torch.zeros_like(test_pred) - 1
    tq = tqdm(testFeatLoader, desc='test', ncols=80, ascii=True)
    for i, (feat, y, psg) in enumerate(tq):
        with torch.no_grad():
            p = model(feat.cuda())
            pred = age(p)
        pred = pred.cpu()
        test_pred[i*BATCH_SIZE:i*BATCH_SIZE+BATCH_SIZE] = pred
        test_y[i*BATCH_SIZE:i*BATCH_SIZE+BATCH_SIZE] = y
        test_psg[i*BATCH_SIZE:i*BATCH_SIZE+BATCH_SIZE] = psg

    # eval end
    mae = torch.abs(test_pred - test_y).mean().item()
    std = torch.abs(test_pred - test_y).std().item()
    r = np.corrcoef(test_pred, test_y)[0,1]
    print('TEST: MAE = %.3f, std = %.3f, R = %.3f'%(mae, std, r))
    # save
    if not os.path.exists(r'experiments\%s\results'%cfg.experiment):
        os.mkdir(r'experiments\%s\results'%cfg.experiment)
    np.savetxt(r'experiments\%s\results\%s_post.csv'%(cfg.experiment, cfg.tvt),
            np.stack([test_psg.reshape(-1), test_y, test_pred]).swapaxes(0,1),fmt='%.1f',delimiter=',')
    return test_pred, test_y


if __name__ == "__main__":
    yaml_path = r'experiments\holdout_hyp\config.yaml'

    yh = YamlHandler(yaml_path) 
    cfg = yh.read_yaml()
    tester_post(cfg)
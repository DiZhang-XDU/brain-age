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


def tester(cfg, withFeat = True):
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.CUDA_VISIBLE_DEVICES
    resultObj = torch.load(cfg.best_ckp)
    model = resultObj['net'].cuda()
    if type(model) == DP:
        model = model.module
    if cfg.eval_parallel:
        model = DP(model)
        
    myDataset = datasets(cfg)
    testSet = myDataset(cfg, tvt=cfg.tvt)
    testLoader = torch.utils.data.DataLoader(testSet, batch_size = 16, 
                    shuffle = False, num_workers = 2, drop_last = False, pin_memory = True)
    model.eval()
    age = MedianAge().cuda()
    age.eval()

    ### init vars
    psg_pred = {}
    psg_target = {}
    psg_feat = {}
    ###

    with torch.no_grad():
        tic = time.time()
        tq = tqdm(testLoader, desc= 'Test', ncols=80, ascii=True)
        for i, (x_eeg, x_hyp, x_loc, target, psg) in enumerate(tq):
            with torch.no_grad(): 
                p, f = model(x_eeg.cuda(), 
                                x_hyp.cuda(), 
                                x_loc.cuda())

            # record
            with torch.no_grad():
                pred = age(p) if cfg.criterion == 'DecadeCE' else p.view(-1)
            pred = pred.cpu()

            feature = f.cpu().detach().clone().view([f.size(0),-1])

            for j in range(len(psg)):
                idx_psg = int(psg[j])
                if idx_psg not in psg_target:
                    psg_pred[idx_psg] = torch.empty([0]).int()
                    psg_target[idx_psg] = int(target[j])
                    psg_feat[idx_psg] = torch.empty([0, feature.size(1)])
                psg_pred[idx_psg] = torch.cat([psg_pred[idx_psg], pred[j:j+1].int()])
                if withFeat:
                    psg_feat[idx_psg] = torch.cat([psg_feat[idx_psg], feature[j:j+1]])


    # performance
    y_true, y_pred, idx = [], [], [] # global vars
    for s in psg_target:
        idx += [[s]] * len(psg_pred[s])
        y_true += [psg_target[s]] * len(psg_pred[s])
        y_pred += psg_pred[s]
    idx = np.array(idx)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    report = 'Task:%s\nTest Time:%.2fs\n'%(cfg.experiment,time.time()-tic)
    report += 'Global...\n'
    report += 'TEST: MAE = %.3f, std = %.3f, R = %.3f\n'%(
        np.abs(y_true - y_pred).mean(),
        np.abs(y_true - y_pred).std(),
        np.corrcoef(y_true, y_pred)[0,1])

    # save
    print('saving ...')
    assert len(psg_pred) == len(psg_target) == len(psg_feat)
    saveResult = './experiments/%s/results'%(cfg.experiment)
    if not os.path.exists(saveResult):
        os.mkdir(saveResult)
    # save pkl
    with open(join(saveResult, '%s.pkl'%cfg.tvt), 'wb') as f:
        pickle.dump({'pred':psg_pred,'target':psg_target}, f)
    if cfg.tvt == 'test' and withFeat:
        with open(saveResult.replace('results', 'prepared_data/test_feature.pkl'), 'wb') as f:
            pickle.dump(psg_feat, f)
    # save report
    with open(join(saveResult, '%s_report.txt'%cfg.tvt), 'w') as f:
        f.write(report)
    # save csv
    np.savetxt(join(saveResult, '%s.csv'%cfg.tvt),
            np.stack([idx.reshape(-1), y_true, y_pred]).swapaxes(0,1),fmt='%.1f',delimiter=',')
    print('done!')
    return psg_pred, psg_target, psg_feat


if __name__ == "__main__":
    yaml_path = r'experiments\swin_merge\config.yaml'

    yh = YamlHandler(yaml_path)
    cfg = yh.read_yaml()
    tester(cfg, withFeat=True)
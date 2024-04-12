import numpy as np
import torch,os,pickle
import time
from tqdm import tqdm
from torch.nn import DataParallel as DP
from mne.io import read_raw_edf

np.set_printoptions(suppress=True) 
import warnings
warnings.filterwarnings("ignore")
import multiprocessing
multiprocessing.set_start_method('spawn', True)
import sys
sys.path.append(os.getcwd())

from tools import *

_EPOCH_SEC_SIZE = 30
_stage_dict = {
    "W": 0,
    "N1": 1,
    "N2": 2,
    "N3": 3,
    "REM": 4,
    "UNKNOWN" :5
}
chnNameTable = {
        'SHHS1':{'C4':'EEG', 'C3':('EEG(sec)', 'EEG2','EEG 2','EEG(SEC)','EEG sec'),
                'E1':'EOG(L)', 'E2':'EOG(R)'},
        'SHHS2':{'C4':'EEG', 'C3':('EEG(sec)', 'EEG2'),
                'E1':'EOG(L)','E2':'EOG(R)'},
        'CFS':{'E1':'LOC', 'E2':'ROC', 'EMG':'EMG2', 'EMGref':'EMG1'},
        'MESA':{'F4':'EEG1','C4':'EEG3','O2':'EEG2','E1':'EOG-L','E2':'EOG-R'},
        'HPAP1':{'C3':'C3-M2', 'C4':'C4-M1',
                'E1':('E1-M2','E-1','L-EOG','LOC','E1-E2'), 
                'E2':('E2-M1','E-2','R-EOG','ROC'),
                'M2':'E2-M1',
                'EMG':('LCHIN','CHIN','CHIN1-CHIN2','Lchin-Cchin','EMG1','L.','Chin1','Chin EMG'),
                'EMGref':('CCHIN','RCHIN','EMG2','C.','Chin2')},
        'WSC':{'F3':('F3_M2','F3_M1','F3_AVG'),'C3':('C3_M2','C3_M1'), 'O1':('O1_M2','O1_M1','O1_AVG'), 
                'F4':'F4_M1','C4':('C4_M1','C4_AVG'), 'O2':'O2_M1',
                'EMG':('chin','cchin_l','rchin_l'),'EMGref':'cchin_r'  },
        'NCHSDB':{'F3':('EEG F3-M2', 'EEG F3'), 'F4':('EEG F4-M1', 'EEG F4'),
                'C3':('EEG C3-M2', 'EEG C3'), 'C4':('EEG C4-M1', 'EEG C4'),
                'O1':('EEG O1-M2', 'EEG O1'), 'O2':('EEG O2-M1', 'EEG O2'),
                'E1':('EOG LOC-M2', 'LOC', 'EEG E1'), 'E2':('EOG ROC-M1', 'ROC', 'EEG E2'),
                'EMG':('EMG Chin1-Chin2', 'EMG Chin2-Chin1', 'EMG Chin1-Chin3', 'EMG Chin3-Chin2', 'EEG Chin1-Chin2', 'Chin1', 'EEG Chin1'),
                'EMGref':('Chin2', 'EEG Chin2')},
        'MROS1':{'C4':'C4-A1','C3':'C3-A2',
                'E1':'LOC', 'E2':'ROC', 'M1':'A1', 'M2':'A2',
                'EMG':('LChin', 'L Chin', 'L Chin-R Chin'),
                'EMGref':('RChin','R Chin')},
        'MROS2':{'C4':'C4-A1','C3':'C3-A2',
                'E1':'LOC', 'E2':'ROC', 'M1':'A1', 'M2':'A2',
                'EMG':('LChin', 'L Chin', 'L Chin-R Chin'),
                'EMGref':('RChin','R Chin')},
        'ABC1':{'EMG':'Chin2', 'EMGref':'Chin1'},
        'ABC2':{'EMG':'Chin2', 'EMGref':'Chin1'},
        'ABC3':{'EMG':'Chin2', 'EMGref':'Chin1'},
}

def _getExpectedChnNames(chnNameTable, rawChns):
    chnNames = {}
    expect = ['F3','F4','C3','C4','O1','O2','E1','E2','EMG','M1','M2','EMGref']
    for c in expect:
        found = False
        names = chnNameTable[c] if c in chnNameTable else []
        names = [names] if type(names) is str else list(names)
        names.append(c) # default channel name
        for name in names:
            for rcn in rawChns:
                if name.upper() == rcn.upper():
                    chnNames[c] = rcn;found = True;break
            if found:break
    # del exist ref
    for ref in ('M1', 'M2'):
        if ref in chnNames:
            for c in ('F3','F4','C3','C4','O1','O2','E1','E2'):
                if (c in chnNames) and (chnNames[ref] in chnNames[c]):
                    del chnNames[ref]
                    break
    # replace alternative channel if ["F4, C4, O2, EMG"] not exist. BUG: replace ref
    if ('F4' not in chnNames) and ('F3' in chnNames):
        chnNames['F4'] = chnNames['F3']
        del chnNames['F3']
    if ('C4' not in chnNames) and ('C3' in chnNames):
        chnNames['C4'] = chnNames['C3']
        del chnNames['C3']
    if ('O2' not in chnNames) and ('O1' in chnNames):
        chnNames['O2'] = chnNames['O1']
        del chnNames['O1']
    if ('EMG' not in chnNames) and ('EMGref' in chnNames):
        chnNames['EMG'] = chnNames['EMGref']
        del chnNames['EMGref']
    return chnNames

def _checkChnValue(data, sampling_rate = 128):
    n_except = 0
    d = data.flatten()
    n_epoch = len(d) // (_EPOCH_SEC_SIZE * sampling_rate)
    for i in range(n_epoch):
        if not 0.2 < np.diff(np.percentile(d[i*_EPOCH_SEC_SIZE*sampling_rate:(i+1)*_EPOCH_SEC_SIZE*sampling_rate], [25, 75])) < 2e2:
            n_except += 1
    return n_except / n_epoch

def data_generator(edfName = '', sampling_rate = 128, channel = None):
    # load head
    raw = read_raw_edf(edfName, preload=False, stim_channel=None)

    # get raw Sample Freq and Channel Name
    sfreq = raw.info['sfreq']
    resample = False if sfreq == sampling_rate else True
    print('【signal sampling freq】:',sfreq)
    ch_names_exist = _getExpectedChnNames(chnNameTable[cfg.dataset] ,raw.ch_names)
    # ch_names ready!

    # load raw signal
    exclude_channel = raw.ch_names
    for cn in set([ch_names_exist[c] for c in ch_names_exist]):
        if cn is not None: 
            exclude_channel.remove(cn)
    raw = read_raw_edf(edfName, preload=True, stim_channel=None, 
                            exclude=exclude_channel)    
    # preprocessing
    X_data = None
    X_order = {'F4':0,'C4':1,'O2':2,'E1':3,'E2':4,'EMG':5}
    ch_name_reverse = {ch_names_exist[c]:c for c in ch_names_exist}
    ch_name_alter = {'F4':'F3','C4':'C3','O2':'O1'}
    # start
    eeg_picks = [ch_names_exist[n] for n in ('F4', 'C4', 'O2', 'M1') if n in ch_names_exist]
    eog_picks = [ch_names_exist[n] for n in ('E1', 'E2', 'M2') if n in ch_names_exist]
    emg_picks = [ch_names_exist[n] for n in ('EMG', 'EMGref') if n in ch_names_exist]
    raw_eeg = raw.copy().pick(eeg_picks)
    raw_eog = raw.copy().pick(eog_picks)
    raw_emg = raw.copy().pick(emg_picks)
    if 'M1' in ch_names_exist:
        raw_eeg.set_eeg_reference([ch_names_exist['M1']])
        raw_eeg = raw_eeg.pick(eeg_picks[:-1])
    if 'M2' in ch_names_exist:
        raw_eog.set_eeg_reference([ch_names_exist['M2']])
        raw_eog = raw_eog.pick(eog_picks[:-1])
    if 'EMGref' in ch_names_exist:
        raw_emg.set_eeg_reference([ch_names_exist['EMGref']])
        raw_emg = raw_emg.pick(emg_picks[:-1])
    for r in (raw_eeg, raw_eog, raw_emg):
        assert len(r.ch_names) > 0
        if r.info['sfreq'] > 120:
            r.notch_filter([50,60])
        if r is not raw_emg:
            r.filter(l_freq = 0.3, h_freq = 35, method='iir')
        else:
            r.filter(l_freq = 10, h_freq = 49, method='iir')
        if resample:
            r.resample(sampling_rate)

        # channel by channel check
        for c in r.ch_names:
            c_data, _ = r[c]    # shape: [1, length]
            ################  Unit: Volt to μV  ################       
            if (r._orig_units[c] in ('µV', 'mV')) or (np.std(c_data) < 1e-3):
                c_data *= 1e6
            #################### Check Unit ####################         
            p = np.percentile(c_data, [5, 25, 75, 95])
            assert 1 < p[2]-p[1] < 1e3 or 5e-2<np.std(c_data)<3e3 or p[3]-p[0] < 1
            ##################### eeg alter ####################
            if (r is raw_eeg) and (ch_name_alter[ch_name_reverse[c]] in ch_names_exist):
                bad_epoch_rate = _checkChnValue(c_data)
                if bad_epoch_rate > 0.1:
                    cname_alter = ch_name_alter[ch_name_reverse[c]]   # F3,C3,O1
                    cname_alter = [ch_names_exist[cname_alter], ch_names_exist['M1']] if 'M1' in ch_names_exist else [ch_names_exist[cname_alter]]
                    r_alter = raw.copy().pick(cname_alter)
                    if 'M1' in ch_names_exist:
                        r_alter.set_eeg_reference([ch_names_exist['M1']])
                        r_alter = r_alter.pick(cname_alter[:-1])
                    if r_alter.info['sfreq'] > 120:
                        r_alter.notch_filter([50,60])
                    r_alter.filter(l_freq = 0.3, h_freq = 35, method='iir')
                    if resample:
                        r_alter.resample(sampling_rate)
                    data_alter, _ = r_alter[cname_alter[0]]
                    if (r_alter._orig_units[cname_alter[0]] in ('µV', 'mV')) or (np.std(data_alter) < 1e-3):
                        data_alter *= 1e6
                    bad_alter_rate = _checkChnValue(data_alter)
                    if bad_alter_rate < bad_epoch_rate:
                        c_data = data_alter
            ####################### Save #######################
            if X_data is None:
                X_data = np.zeros([c_data.shape[1], 6])
            X_data[:, X_order[ch_name_reverse[c]]] = c_data
    # X_data shape: [length, chn]
    n_epochs = len(X_data) // (_EPOCH_SEC_SIZE * sampling_rate)
    X_data_slim = X_data[:int(n_epochs * _EPOCH_SEC_SIZE * sampling_rate)]
    X = np.asarray(np.split(X_data_slim, n_epochs)).astype(np.float32)
    X = torch.from_numpy(X).float()
    X = torch.clip(X, -1000, 1000)
    X = X[:,:,:3] # Select 3 EEG
    # X ready
    return X

def tester(cfg):
    # prepare model
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.CUDA_VISIBLE_DEVICES
    model_epoch = torch.load(cfg.best_ckp)['net'].cuda()
    model_post = torch.load(cfg.post_ckp)['net'].cuda()
    if type(model_epoch) == DP:
        model_epoch = model_epoch.module
    if type(model_post) == DP:
        model_post = model_post.module
    model_epoch.eval()
    model_post.eval()
    age = MedianAge().cuda()
    age.eval()

    # prepare input
    X = data_generator(cfg.edf) # [n_epoch, len_epoch, n_channel]
    with open('data/subjects/%s/%s.info'%(cfg.psgID, cfg.psgID), 'r') as f:
        chronological_age = float(f.readline().strip().split(',')[-1])
    with open('data/subjects/%s/pred_hypnodensity.pkl'%cfg.psgID, 'rb') as f:
        hyp = pickle.load(f)[:,:5].softmax(1)
        # set hypnodensity length to 1260
        if len(hyp) <= 1260:
            hyp_scaled = torch.tensor([1., 0., 0., 0., 0.]).repeat(1260, 1)
            hyp_scaled[:len(hyp)] = hyp
        else:
            hyp_scaled = hyp[:1260]
        hyp = hyp_scaled
            
    # eval
    with torch.no_grad():
        tic = time.time()
        # overlap = 50%
        tq = tqdm(range(0, len(X) - 10, 5), desc= 'Test', ncols=80, ascii=True)  
        result_epoch = torch.empty([0]).float()
        feature_epoch = torch.empty([0, 48 + 10]).float()

        # epoch estimation & deep feature
        for i in tq:
            input_eeg = X[i: i + 10].view(-1, X.size(-1)).swapaxes(0,1).unsqueeze(0)  # B, C, L
            input_hyp = hyp.swapaxes(0,1).unsqueeze(0)  # B, C, L
            p, f = model_epoch(input_eeg.cuda(),        # 5min eeg epoch
                            input_hyp.cuda(),           # psg hypnodensity
                            torch.empty([1]).cuda())    # nevermind
            pred = age(p)

            # result
            pred = pred.cpu()         
            feature = f.cpu().detach().clone().view([f.size(0),-1])

            result_epoch = torch.cat([result_epoch, pred])
            feature_epoch = torch.cat([feature_epoch, feature])

        # post estimation
        psg_len = 252
        if len(feature_epoch) < psg_len:
            feature_epoch = torch.cat([feature_epoch, torch.zeros([psg_len - len(feature_epoch), feature_epoch.size(1)])]).float()
        else:
            feature_epoch = feature_epoch[:psg_len]
            
        p = model_post(feature_epoch.swapaxes(0,1).unsqueeze(0).cuda())   # B, C, L
        pred_hyp = age(p).cpu()

        # output
        print('Dataset: %s\nEDF: %s\nCA = %.1fyr\nBA_5min = %.1f±%.1fyr\nBA = %.1fyr'%\
              (cfg.dataset, cfg.edf, chronological_age, result_epoch.mean(), result_epoch.std(), pred_hyp))

    return chronological_age, result_epoch, pred_hyp


if __name__ == "__main__":
    from tools.config_handler import yamlStruct
    # pars
    cfg = yamlStruct()
    cfg.add('CUDA_VISIBLE_DEVICES','0')
    cfg.add('edf',r'\\192.168.31.100\Data\13_公开数据集\公开数据集\nsrr\homepap\polysomnography\edfs\lab\full\homepap-lab-full-1600001.edf')
    cfg.add('psgID','015894')
    cfg.add('dataset','HPAP1')

    cfg.add('best_ckp',r"experiments\swin_merge\weights\best_checkpoint")
    cfg.add('post_ckp',r"experiments\swin_merge\weights\post_checkpoint")
    
    # inference
    ca, ba_, ba = tester(cfg)

    # plot
    import matplotlib.pyplot as plt
    x = range(len(ba_))
    plt.plot([0, len(ba_)], [ca, ca], label = 'CA')
    plt.plot([0, len(ba_)], [ba, ba], label = 'BA')
    plt.plot(x, ba_, label = 'BA_5min')
    plt.legend()
    plt.show()
    print('done')
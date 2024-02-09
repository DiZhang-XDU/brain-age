import os,time,pickle
import numpy as np
import torch
from timm import scheduler as schedulers    # do not delete!

from os.path import join
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.nn import DataParallel as DP
from tools import *
from sleep_models import build_model

import warnings
warnings.filterwarnings("ignore")##################################
import multiprocessing
multiprocessing.set_start_method('spawn', True)

def trainer(cfg):
    # environment
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.CUDA_VISIBLE_DEVICES
    for p in [cfg.path.weights, cfg.path.tblogs]:
        if os.path.exists(p) is False:
            os.mkdir(p)
    seed = 0 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    criterion = criterions(cfg).cuda()
    earlyStop = 0 # for huber loss

    # tensorboard
    import shutil
    if (not cfg.resume) and os.path.exists(cfg.path.tblogs):
        shutil.rmtree(cfg.path.tblogs)
    writer = SummaryWriter(cfg.path.tblogs)

    # prepare dataloader 
    myDataset = datasets(cfg)
    trainSet = myDataset(cfg, tvt = 'train')
    validSet = myDataset(cfg, tvt = 'valid')
    trainLoader = torch.utils.data.DataLoader(trainSet, batch_size = cfg.BATCH_SIZE, persistent_workers=True,
                                    shuffle = True, num_workers = cfg.train_thread, drop_last = False, pin_memory = True)
    validLoader = torch.utils.data.DataLoader(validSet, batch_size = cfg.BATCH_SIZE, persistent_workers=True,
                                    shuffle = True, num_workers = cfg.eval_thread, drop_last = False, pin_memory = True)

    # model initialization
    model = build_model(cfg).cuda()
    if cfg.resume:
        optim = torch.optim.SGD(model.parameters(), lr= 1e-1, momentum = .9, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim,4,1e-3)
        loadObj = torch.load(join(cfg.path.weights, 'checkpoint_'), map_location='cpu')
        model_, epoch_, optim_, scheduler_, best_loss_val_ = loadObj['net'], loadObj['epoch'], loadObj['optim'], loadObj['sched'], loadObj['best_loss_val']
        model.load_state_dict(model_.state_dict())
        optim.load_state_dict(optim_.state_dict())
        best_loss_val, best_val_mae, epoch = 9999, 9999, 0
    else:
        optim = eval(cfg.optimizer)
        scheduler = eval(cfg.scheduler)
        best_loss_val, best_val_mae, epoch = 9999, 9999, 0
    if cfg.train_parallel:
        model = DP(model)
    age = MedianAge().cuda()
    augment = DataAugment().cuda()
    age.eval()
    augment.eval()

    print('start epoch')
    step = 0
    trainIter = iter(trainLoader)   
    for epoch in range(epoch, cfg.EPOCH_MAX): 
        tic = time.time()
        name = ('train', 'valid')
        epoch_loss = {i:0 for i in name}
        epoch_mae = {i:0 for i in name}

        record_target = {'train':torch.zeros(cfg.EPOCH_STEP * cfg.BATCH_SIZE) - 1, 
                        'valid':torch.zeros(len(validSet)) - 1}
        record_pred = {'train':torch.zeros(cfg.EPOCH_STEP * cfg.BATCH_SIZE) - 1, 
                        'valid':torch.zeros(len(validSet)) - 1}

        torch.cuda.empty_cache()
        model.train()
        tq = tqdm(range(cfg.EPOCH_STEP), desc= 'Trn', ncols=80, ascii=True)
        # with torch.autograd.profiler.profile(enabled=True) as prof:
        for i, _ in enumerate(tq):
            x_eeg, x_hyp, x_loc, target, _ = next(trainIter)
            step += 1
            if step == len(trainLoader):    # recurrent loader
                step = 0
                trainIter = iter(trainLoader)
            if epoch == 0 and i == int(0.1*cfg.EPOCH_STEP) + 1: # warm up
                scheduler.step_update((epoch + 1) * cfg.EPOCH_STEP)
        
            # data augment in training
            x_eeg = x_eeg.cuda()
            with torch.no_grad():
                x_eeg = augment(x_eeg)
            
            # forward
            p, _ = model(x_eeg, 
                        x_hyp.cuda(), 
                        x_loc.cuda(), 
                        # alpha = 0.5*(epoch/cfg.EPOCH_MAX)**2)
                        alpha = 0.1)
            loss = criterion(p, target.cuda())
            if cfg.train_parallel:
                loss = torch.mean(loss)
            with torch.no_grad():
                pred = age(p) if cfg.criterion == 'DecadeCE' else p.view(-1)
            
            # backward
            optim.zero_grad()
            loss.backward()
            optim.step()

            # record
            pred = pred.cpu()
            record_pred['train'][i*cfg.BATCH_SIZE:i*cfg.BATCH_SIZE+pred.shape[0]] = pred
            record_target['train'][i*cfg.BATCH_SIZE:i*cfg.BATCH_SIZE+pred.shape[0]] = target

            epoch_loss['train'] += loss.item()
            epoch_mae['train'] += torch.abs(pred - target).mean().item()

            tq.set_postfix({'Loss':'{:.4f}'.format(epoch_loss['train'] / (tq.n+1)), 
                            'MAE:':'{:.4f}'.format(epoch_mae['train'] / (i+1))})
        # print(prof.key_averages().table(sort_by='self_cpu_time_total'))
        epoch_loss['train'] /= (i+1)
        epoch_mae['train'] /= (i+1)

        # eval
        torch.cuda.empty_cache()
        model_eval = DP(model) if cfg.eval_parallel and not cfg.train_parallel else model
        model_eval.eval()
        
        with torch.no_grad():
            tq = tqdm(validLoader, desc = 'Val', ncols=75, ascii=True)
            for i, (x_eeg, x_hyp, x_loc, target, _) in enumerate(tq):
                with torch.no_grad(): 
                    p, _ = model_eval(x_eeg.cuda(), 
                                    x_hyp.cuda(), 
                                    x_loc.cuda())
                    pred = age(p) if cfg.criterion == 'DecadeCE' else p.view(-1)
                loss = criterion(p, target.cuda())
                if cfg.eval_parallel:
                    loss = torch.mean(loss)
                
                # record
                pred = pred.cpu()
                record_pred['valid'][i*cfg.BATCH_SIZE:i*cfg.BATCH_SIZE+cfg.BATCH_SIZE] = pred
                record_target['valid'][i*cfg.BATCH_SIZE:i*cfg.BATCH_SIZE+cfg.BATCH_SIZE] = target

                epoch_loss['valid'] += loss.item()
                epoch_mae['valid'] += torch.abs(target - pred).mean().item()
                
                tq.set_postfix({'Loss':'{:.4f}'.format(epoch_loss['valid'] / (i+1)), 
                            'MAE:':'{:.4f}'.format(epoch_mae['valid'] / (i+1))})
                        
        epoch_loss['valid'] /= (i+1)
        
        # epoch end now.
        scheduler.step_update((epoch + 1) * cfg.EPOCH_STEP)
        
        # epoch stat
        record_pred['train'] = record_pred['train'][record_pred['train'] != -1]
        record_target['train'] = record_target['train'][record_target['train'] != -1]
        assert len(record_pred['train']) == len(record_target['train'])

        for idx in name:
            epoch_mae[idx] = torch.abs(record_target[idx] - record_pred[idx]).mean()

        msg_epoch = 'epoch:{:02d}, time:{:2f}\n'.format(epoch, time.time() - tic)
        msg_loss = 'Trn Loss:{:.4f}, mae:{:.2f}  Val Loss:{:.4f}, mae:{:.2f}\n'.format(
            epoch_loss['train'], epoch_mae['train'], epoch_loss['valid'], epoch_mae['valid'])

        print(msg_epoch + msg_loss + '\n')

        # save
        writer.add_scalars('Loss',{'train':epoch_loss['train'] , 'valid':epoch_loss['valid']},epoch)
        writer.add_scalars('MAE',{'train':epoch_mae['train'], 'valid':epoch_mae['valid']},epoch)

        with open(cfg.path.log, 'a') as f:
            f.write(msg_epoch)
            f.write(msg_loss)
        
        if best_val_mae > epoch_mae['valid']:
            earlyStop = -1
            best_val_mae = epoch_mae['valid']
            saveObj = {'net': model, 'epoch':epoch, 'optim':optim , 'sched':scheduler, 'best_loss_val':best_loss_val}
            torch.save(saveObj, join(cfg.path.weights, 'best_checkpoint'))
        torch.save(saveObj, join(cfg.path.weights, 'epoch_{:02d}_val_loss={:4f}_mae={:.4f}'.format(epoch, epoch_loss['valid'], epoch_mae['valid'])))
        
        # early stop when using abrinkk param
        earlyStop+=1
        if earlyStop == 3 and cfg.criterion == 'huber':    
            break

    writer.close()

if __name__ == "__main__":
    yaml_path = r'experiments\swin_merge\config.yaml'

    yh = YamlHandler(yaml_path)
    cfg = yh.read_yaml()
    trainer(cfg)
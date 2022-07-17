import argparse
import shutil
from datetime import datetime

import yaml
import torch
from torch import nn
import torch.distributed as dist
from prompt_toolkit import prompt
from tqdm import tqdm
import numpy as np
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix

# noinspection PyUnresolvedReferences
from dataset.pipa import Annotations  # legacy to correctly load dataset.
from helper import Helper
from utils.utils import *

# logger = logging.getLogger('logger').setLevel(logging.WARNING)
logger = logging.getLogger('PIL').setLevel(logging.WARNING)


def train(hlpr: Helper, epoch, model, optimizer, train_loader, scaler=None, attack=True):
    losses = []
    criterion = hlpr.task.criterion
    model.train()
    if hlpr.params.dist == True:
            train_loader.sampler.set_epoch(epoch)
    for i, data in tqdm(enumerate(train_loader)):
        batch = hlpr.task.get_batch(i, data)
        model.zero_grad()
        loss = hlpr.attack.compute_blind_loss(model, criterion, batch, attack)
        losses.append(loss.item())
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), 0.1)
            optimizer.step()
        
        if hlpr.params.max_batch_id is not None and i == hlpr.params.max_batch_id:
            break
    
    # print(f"Loss average over epoch {epoch}: {sum(losses)/len(losses)}")
    return sum(losses)/len(losses)


def test(hlpr: Helper, epoch, backdoor=False):
    model = hlpr.task.model
    model.eval()
    hlpr.task.reset_metrics()

    all_preds, all_labels = [], []
    num_correct = 0
    num_samples = 0

    with torch.no_grad():
        for i, data in tqdm(enumerate(hlpr.task.test_loader)):
            batch = hlpr.task.get_batch(i, data)
            if backdoor:
                batch = hlpr.attack.synthesizer.make_backdoor_batch(batch,
                                                                    test=True,
                                                                    attack=True)
            outputs = model(batch.inputs)
            if hlpr.params.loss == 'cross_entropy':
                outputs = outputs.max(1, keepdim=True)[1]
            else:
                outputs[outputs < 0.5] = 0
                outputs[(outputs >= 0.5) & (outputs < 1.5)] = 1
                outputs[(outputs >= 1.5) & (outputs < 2.5)] = 2
                outputs[(outputs >= 2.5) & (outputs < 3.5)] = 3
                outputs[(outputs >= 3.5) & (outputs < 4.5)] = 4
                outputs[(outputs >= 4.5) & (outputs < 100)] = 5
                # outputs[(outputs >= 5.5) & (outputs < 6.5)] = 6
                # outputs[(outputs >= 6.5) & (outputs < 100)] = 7
            
            outputs = outputs.long().view(-1)
            y = batch.labels.long().view(-1)
            num_correct += (outputs == y).sum()
            num_samples += outputs.shape[0]

            # add to lists
            all_preds.append(outputs.detach().cpu().numpy())
            all_labels.append(y.detach().cpu().numpy())
            
            # if i == hlpr.params.max_batch_id // 10:
            #     break
    print(
        f"Got over epoch {epoch}: {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}"
    )
    model.train()
    preds, labels = np.concatenate(all_preds, axis=0).astype(np.int64), np.concatenate(all_labels, axis=0).astype(np.int64)
    if not backdoor:
        print(f"Kappa (Val) over epoch {epoch}, no backdoor: {cohen_kappa_score(labels, preds, weights='quadratic')}")
        print(confusion_matrix(labels, preds))

    return cohen_kappa_score(labels, preds, weights='quadratic')


def run(hlpr):
    if hlpr.params.amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
    print('Training with attack == ', str(hlpr.params.attack))
    for epoch in range(hlpr.params.start_epoch,
                       hlpr.params.epochs + 1):
        loss = train(hlpr, epoch, hlpr.task.model, hlpr.task.optimizer,
              hlpr.task.train_loader, scaler=scaler, attack=hlpr.params.attack)
        print(f"Loss average over epoch {epoch}: {loss}")
        if dist.get_rank() == 0: 
            kappa = test(hlpr, epoch, backdoor=False)
            _ = test(hlpr, epoch, backdoor=True)
            hlpr.save_model(hlpr.task.model, epoch, val_loss=kappa)
        if hlpr.task.scheduler is not None:
            hlpr.task.scheduler.step(epoch)


def fl_run(hlpr: Helper):
    for epoch in range(hlpr.params.start_epoch,
                       hlpr.params.epochs + 1):
        run_fl_round(hlpr, epoch)
        kappa = test(hlpr, epoch, backdoor=False)
        _ = test(hlpr, epoch, backdoor=True)

        hlpr.save_model(hlpr.task.model, epoch, val_loss=kappa)


def run_fl_round(hlpr, epoch):
    global_model = hlpr.task.model
    local_model = hlpr.task.local_model

    round_participants = hlpr.task.sample_users_for_round(epoch)
    weight_accumulator = hlpr.task.get_empty_accumulator()

    for n, user in tqdm(enumerate(round_participants)):
        # print('local data length: ', len(user.train_loader))
        hlpr.task.copy_params(global_model, local_model)
        optimizer = hlpr.task.make_optimizer(local_model)
        losses = []
        for local_epoch in range(hlpr.params.fl_local_epochs):
            if user.compromised: # use backdoor input
                loss = train(hlpr, local_epoch, local_model, optimizer,
                      user.train_loader, attack=True)
                
            else:
                loss = train(hlpr, local_epoch, local_model, optimizer,
                      user.train_loader, attack=False)
            losses.append(loss)
        
        print(f"Loss average over user {n}: {sum(losses) / len(losses)}")
        local_update = hlpr.task.get_fl_update(local_model, global_model)
        if user.compromised:
            hlpr.attack.fl_scale_update(local_update)
        hlpr.task.accumulate_weights(weight_accumulator, local_update)

    hlpr.task.update_global_model(weight_accumulator, global_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Backdoors')
    parser.add_argument('--params', dest='params', default='utils/params.yaml')
    parser.add_argument('--name', dest='name', required=True)
    args = parser.parse_args()

    with open(args.params) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    params['current_time'] = datetime.now().strftime('%b.%d_%H.%M.%S')
    params['name'] = args.name

    if params['dist'] == True:
        torch.backends.cudnn.benchmark = True
        def init_dist(backend='nccl', **kwargs):
            ''' initialization for distributed training'''
            rank = int(os.environ['RANK'])
            num_gpus = torch.cuda.device_count()
            torch.cuda.set_device(rank % num_gpus)
            dist.init_process_group(backend=backend, **kwargs)
        init_dist()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
    
    
    helper = Helper(params)
    if helper.params.fl:
        fl_run(helper)
    else:
        run(helper)

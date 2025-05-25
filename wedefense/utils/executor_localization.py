# Copyright (c) 2021 Hongji Wang (jijijiang77@gmail.com)
#               2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
#               2025 Lin Zhang (partialspoof@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tableprint as tp

import torch
import torchnet as tnt
from wedefense.dataset.dataset_utils import apply_cmvn, spec_aug

##TODO add
#def lab_in_scale(self, labvec, shift):
#    num_frames = int(len(labvec) / shift)
#    if(num_frames==0):
#        #for some case that duration < frameshift. 20211031. the case in 0.64 scale. 
#        num_frames=1
#        new_lab = np.zeros(num_frames, dtype=int)
#        new_lab[0] = min(labvec) #only for binary '0' for spoof
#    else:
#        #common case.
#        new_lab = np.zeros(num_frames, dtype=int)
#        for idx in np.arange(num_frames):
#            st, et  = int(idx * shift), int((idx+1)*shift)
#            new_lab[idx] = min(labvec[st:et]) #only for binary '0' for spoof
#    return new_lab

def run_epoch(dataloader, epoch_iter, model, criterion, optimizer, scheduler,
              margin_scheduler, epoch, logger, scaler, device, configs):
    model.train()
    # By default use average pooling
    loss_meter = tnt.meter.AverageValueMeter()
    acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)

    frontend_type = configs['dataset_args'].get('frontend', 'fbank')
    for i, batch in enumerate(dataloader):
        cur_iter = (epoch - 1) * epoch_iter + i
        scheduler.step(cur_iter)
        margin_scheduler.step(cur_iter)

        utts = batch['key']
        targets = batch['label']
        targets = targets.long().to(device)  # (B, T) T here depends on the label.
        if frontend_type == 'fbank':
            features = batch['feat']  # (B,T,F)
            features = features.float().to(device)
        else:  # 's3prl'
            wavs = batch['wav']  # (B,1,W)
            wavs = wavs.squeeze(1).float().to(device)  # (B,W)
            wavs_len = torch.LongTensor([wavs.shape[1]]).repeat(
                wavs.shape[0]).to(device)  # (B)
            with torch.cuda.amp.autocast(enabled=configs['enable_amp']):
                features, _ = model.module.frontend(wavs, wavs_len) # (B, T, D)

        with torch.cuda.amp.autocast(enabled=configs['enable_amp']):
            # apply cmvn
            if configs['dataset_args'].get('cmvn', True):
                features = apply_cmvn(
                    features, **configs['dataset_args'].get('cmvn_args', {}))
            # spec augmentation
            if configs['dataset_args'].get('spec_aug', False):
                features = spec_aug(features,
                                    **configs['dataset_args']['spec_aug_args'])

            outputs = model(features)  # (B,T,D)  
            embeds = outputs[-1] if isinstance(outputs, tuple) else outputs
            # flatten to (batch_size * seq_length, feat_dim): 
            batch_size = embeds.shape[0]
            feat_dim = embeds.shape[-1]
            outputs = model.module.projection(embeds.view(-1, feat_dim), targets) #(BxT,num_class)
            outputs = outputs.view(targets.shape[0], targets.shape[1] , -1) #->(B, T, num_class)

            if isinstance(outputs, tuple):
                outputs, loss = outputs
            else:
                if isinstance(criterion, torch.nn.MSELoss):
                    # For MSE, we need to convert targets to one-hot
                    targets_one_hot = torch.zeros_like(outputs)
                    targets_one_hot.scatter_(2, targets.unsqueeze(2), 1)  # (B, T, 2)
                    loss = criterion(outputs, targets_one_hot)
                else:
                    loss = criterion(outputs, targets.unsqueeze(2)) #targets: (B, T) -> (B, T, 1)


        # loss, acc
        loss_meter.add(loss.item())
        # acc_meter.add(outputs.cpu().detach().numpy(), targets.cpu().numpy())
        if isinstance(criterion, torch.nn.MSELoss):
            # For MSE loss, using 0.5 as threshold
            preds = (outputs > 0.5).float()
            preds = preds.argmax(dim=2)  # (B, T)
        else:
            # For CE loss, using argmax
            preds = outputs.argmax(dim=2)  # (B, T)
        acc_meter.add(preds.cpu().detach().numpy(), targets.cpu().numpy())

        # updata the model
        optimizer.zero_grad()
        # scaler does nothing here if enable_amp=False
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # log
        if (i + 1) % configs['log_batch_interval'] == 0:
            logger.info(
                tp.row((epoch, i + 1, scheduler.get_lr(),
                        margin_scheduler.get_margin()) +
                       (loss_meter.value()[0], acc_meter.value()[0]),
                       width=10,
                       style='grid'))

        if (i + 1) == epoch_iter:
            break

    logger.info(
        tp.row(
            (epoch, i + 1, scheduler.get_lr(), margin_scheduler.get_margin()) +
            (loss_meter.value()[0], acc_meter.value()[0]),
            width=10,
            style='grid'))


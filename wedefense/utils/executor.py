# Copyright (c) 2021 Hongji Wang (jijijiang77@gmail.com)
#               2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
#               2025 Lin Zhang (partialspoof@gmail.com)
#                    Shuai Wang (wsstriving@gmail.com)
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


def train_epoch(dataloader, epoch_iter, model, criterion, optimizer, scheduler,
                margin_scheduler, epoch, logger, scaler, device, configs):
    """Train the model for one epoch.

    Args:
        dataloader: Training dataloader
        epoch_iter: Number of iterations per epoch
        model: Model to train
        criterion: Loss criterion
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        margin_scheduler: Margin scheduler
        epoch: Current epoch number
        logger: Logger
        scaler: Gradient scaler for mixed precision training
        device: Device to run training on
        configs: Configuration dictionary

    Returns:
        tuple: (training loss, training accuracy, last batch index)
    """
    model.train()
    # By default use average pooling
    loss_meter = tnt.meter.AverageValueMeter()
    acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)

    frontend_type = configs['dataset_args'].get('frontend', 'fbank')
    for i, batch in enumerate(dataloader):
        cur_iter = (epoch - 1) * epoch_iter + i
        scheduler.step(cur_iter)
        margin_scheduler.step(cur_iter)

        targets = batch['label']
        targets = targets.long().to(device)  # (B)
        if frontend_type == 'fbank' or frontend_type.startswith('lfcc'):
            features = batch['feat']  # (B,T,F)
            features = features.float().to(device)
        else:  # 's3prl'
            wavs = batch['wav']  # (B,1,W)
            wavs = wavs.squeeze(1).float().to(device)  # (B,W)
            wavs_len = torch.LongTensor([wavs.shape[1]]).repeat(
                wavs.shape[0]).to(device)  # (B)
            with torch.cuda.amp.autocast(enabled=configs['enable_amp']):
                features, _ = model.module.frontend(wavs, wavs_len)

        with torch.cuda.amp.autocast(enabled=configs['enable_amp']):
            # apply cmvn
            if configs['dataset_args'].get('cmvn', True):
                features = apply_cmvn(
                    features, **configs['dataset_args'].get('cmvn_args', {}))
            # spec augmentation
            if configs['dataset_args'].get('spec_aug', False):
                features = spec_aug(features,
                                    **configs['dataset_args']['spec_aug_args'])

            outputs = model(features)  # (embed_a,embed_b) in most cases
            embeds = outputs[-1] if isinstance(outputs, tuple) else outputs
            outputs = model.module.projection(embeds, targets)
            if isinstance(outputs, tuple):
                outputs, loss = outputs
            else:
                loss = criterion(outputs, targets)

        # loss, acc
        loss_meter.add(loss.item())
        acc_meter.add(outputs.cpu().detach().numpy(), targets.cpu().numpy())

        # update the model
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

    # Final log for this epoch
    logger.info(
        tp.row(
            (epoch, i + 1, scheduler.get_lr(), margin_scheduler.get_margin()) +
            (loss_meter.value()[0], acc_meter.value()[0]),
            width=10,
            style='grid'))


def val_epoch(val_dataloader, val_iter, model, criterion, device, configs):
    """Validate the model on the validation set.

    Args:
        val_dataloader: Validation dataloader
        model: Model to validate
        criterion: Loss criterion
        device: Device to run validation on
        configs: Configuration dictionary

    Returns:
        tuple: (validation loss, validation accuracy)
    """
    model.eval()
    val_loss_meter = tnt.meter.AverageValueMeter()
    val_acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)

    frontend_type = configs['dataset_args'].get('frontend', 'fbank')
    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            targets = batch['label']
            targets = targets.long().to(device)  # (B)
            if frontend_type == 'fbank' or frontend_type.startswith('lfcc'):
                features = batch['feat']  # (B,T,F)
                features = features.float().to(device)
            else:  # 's3prl'
                wavs = batch['wav']  # (B,1,W)
                wavs = wavs.squeeze(1).float().to(device)  # (B,W)
                wavs_len = torch.LongTensor([wavs.shape[1]]).repeat(
                    wavs.shape[0]).to(device)  # (B)
                with torch.cuda.amp.autocast(enabled=configs['enable_amp']):
                    features, _ = model.module.frontend(wavs, wavs_len)

            with torch.cuda.amp.autocast(enabled=configs['enable_amp']):
                # apply cmvn
                if configs['dataset_args'].get('cmvn', True):
                    features = apply_cmvn(
                        features, **configs['dataset_args'].get('cmvn_args', {}))

                outputs = model(features)  # (embed_a,embed_b) in most cases
                embeds = outputs[-1] if isinstance(outputs, tuple) else outputs
                outputs = model.module.projection(embeds, targets)
                if isinstance(outputs, tuple):
                    outputs, loss = outputs
                else:
                    loss = criterion(outputs, targets)

            # loss, acc
            val_loss_meter.add(loss.item())
            val_acc_meter.add(outputs.cpu().detach().numpy(), targets.cpu().numpy())
            if (i + 1) == val_iter:
                break

    return val_loss_meter.value()[0], val_acc_meter.value()[0]

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

# TODO add
# def lab_in_scale(self, labvec, shift):
#    num_frames = int(len(labvec) / shift)
#    if(num_frames==0):
#        # for some case that duration < frameshift.
#        # 20211031. the case in 0.64 scale.
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


def train_epoch(dataloader,
                epoch_iter,
                model,
                criterion,
                optimizer,
                scheduler,
                margin_scheduler,
                epoch,
                logger,
                scaler,
                device,
                configs,
                wandb_log=None):
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
    acc_meter = tnt.meter.acc_meter = tnt.meter.AverageValueMeter()

    frontend_type = configs['dataset_args'].get('frontend', 'fbank')
    for i, batch in enumerate(dataloader):
        cur_iter = (epoch - 1) * epoch_iter + i
        scheduler.step(cur_iter)
        margin_scheduler.step(cur_iter)

        utts = batch['key']
        targets = batch['label']
        targets = targets.long().to(
            device)  # (B, T) T here depends on the label.
        if frontend_type == 'fbank':
            features = batch['feat']  # (B,T,F)
            features = features.float().to(device)
        else:  # 's3prl'
            wavs = batch['wav']  # (B,1,W)
            wavs = wavs.squeeze(1).float().to(device)  # (B,W)
            wavs_len = torch.LongTensor([wavs.shape[1]]).repeat(
                wavs.shape[0]).to(device)  # (B)
            with torch.cuda.amp.autocast(enabled=configs['enable_amp']):
                features, _ = model.module.frontend(wavs,
                                                    wavs_len)  # (B, T, D)

        with torch.cuda.amp.autocast(enabled=configs['enable_amp']):
            # apply cmvn
            if configs['dataset_args'].get('cmvn', True):
                features = apply_cmvn(
                    features, **configs['dataset_args'].get('cmvn_args', {}))
            # spec augmentation
            if configs['dataset_args'].get('spec_aug', False):
                features = spec_aug(features,
                                    **configs['dataset_args']['spec_aug_args'])

            num_class = configs['projection_args']['num_class']
            if hasattr(model, 'module') and hasattr(model.module,
                                                    'get_frame_emb'):
                outputs = model.module.get_frame_emb(features)
            else:
                outputs = model(features)  # (B,T,D)
            embeds = outputs[-1] if isinstance(outputs, tuple) else outputs
            # flatten to (batch_size * seq_length, feat_dim):
            batch_size = embeds.shape[0]
            feat_dim = embeds.shape[-1]
            outputs = model.module.projection(embeds.view(-1, feat_dim),
                                              targets)  # (BxT,num_class)
            outputs = outputs.view(targets.shape[0], targets.shape[1],
                                   -1)  # ->(B, T, num_class)

            if isinstance(outputs, tuple):
                outputs, loss = outputs
            else:
                if isinstance(criterion, torch.nn.MSELoss):
                    # For MSE, we need to convert targets to one-hot
                    targets_one_hot = torch.zeros_like(outputs)
                    targets_one_hot.scatter_(2, targets.unsqueeze(2),
                                             1)  # (B, T, 2)
                    loss = criterion(outputs, targets_one_hot)
                else:
                    # outputs: (B, T, num_class = 2) -> (B*T, num_class)
                    # targets: (B, T) -> (B*T)
                    loss = criterion(outputs.view(-1, num_class),
                                     targets.view(-1))

        # loss, acc
        loss_meter.add(loss.item())
        # acc_meter.add(outputs.cpu().detach().numpy(), targets.cpu().numpy())
        preds = outputs.argmax(dim=2)
        correct = (preds == targets).float().sum()
        total = len(targets.view(-1))
        accuracy = (correct / total).item()
        acc_meter.add(accuracy)

        # updata the model
        optimizer.zero_grad()
        # scaler does nothing here if enable_amp=False
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if (wandb_log):
            wandb_log.log({
                "learning_rate": scheduler.get_lr(),
                "train/loss": loss_meter.value()[0],
                "train/acc": acc_meter.value()[0]
            })

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


def val_epoch(val_dataloader,
              val_iter,
              model,
              criterion,
              device,
              configs,
              wandb_log=None):
    """Validate the model on the validation set for localization task.

    Args:
        val_dataloader: Validation dataloader
        val_iter: Number of iterations in validation
        model: Model to validate
        criterion: Loss criterion
        device: Device to run validation on
        configs: Configuration dictionary
        wandb_log: Optional wandb logger

    Returns:
        tuple: (validation loss, validation accuracy)
    """
    model.eval()
    import torchnet as tnt
    val_loss_meter = tnt.meter.AverageValueMeter()
    val_acc_meter = tnt.meter.AverageValueMeter()

    num_class = 2  # for localization
    frontend_type = configs['dataset_args'].get('frontend', 'fbank')

    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            targets = batch['label']  # (B, T)
            targets = targets.long().to(device)
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
                    from wedefense.dataset.dataset_utils import apply_cmvn
                    features = apply_cmvn(
                        features,
                        **configs['dataset_args'].get('cmvn_args', {}))
                # spec augmentation (usually not for val, but keep for symmetry)
                if configs['dataset_args'].get('spec_aug', False):
                    from wedefense.dataset.dataset_utils import spec_aug
                    features = spec_aug(
                        features, **configs['dataset_args']['spec_aug_args'])
                outputs = model(features)  # (B,T,num_class)
                # criterion expects outputs shape appropriate for sample
                if criterion.__class__.__name__ == "BCEWithLogitsLoss":
                    import torch.nn.functional as F
                    targets_one_hot = F.one_hot(targets,
                                                num_classes=num_class).float()
                    loss = criterion(outputs, targets_one_hot)
                else:
                    # outputs: (B,T,num_class) -> (B*T, num_class)
                    # targets: (B,T) -> (B*T)
                    loss = criterion(outputs.view(-1, num_class),
                                     targets.view(-1))
            val_loss_meter.add(loss.item())

            preds = outputs.argmax(dim=2)
            correct = (preds == targets).float().sum()
            total = len(targets.view(-1))
            accuracy = (correct / total).item()
            val_acc_meter.add(accuracy)

            if wandb_log:
                wandb_log.log({
                    "val/loss": val_loss_meter.value()[0],
                    "val/acc": val_acc_meter.value()[0]
                })
            if (i + 1) == val_iter:
                break

    return val_loss_meter.value()[0], val_acc_meter.value()[0]

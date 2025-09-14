# Copyright (c) 2021 Hongji Wang (jijijiang77@gmail.com)
#               2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
#               2025 Lin Zhang (partialspoof@gmail.com)
#                    Shuai Wang (wsstriving@gmail.com)
#                    Junyi Peng (pengjy@fit.vut.cz)
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
from wedefense.utils.prune_utils import pruning_loss, get_progressive_sparsity


def train_epoch(dataloader,
                epoch_iter,
                model,
                criterion,
                optimizers,
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
        optimizers: A tuple containing the main optimizer and regularization optimizer.
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

    optimizer, optimizer_reg = optimizers

    # By default use average pooling
    loss_meter = tnt.meter.AverageValueMeter()
    acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)

    # Initialize loss meters
    cls_loss_meter = tnt.meter.AverageValueMeter() 
    pruning_loss_meter = tnt.meter.AverageValueMeter()
    
    # Pruning configuration
    use_pruning = configs.get('use_pruning_loss', False) 
    if use_pruning:
        target_sp = configs.get('target_sparsity', 0.5)
        l1, l2 = configs.get('lambda_pair', (1.0, 5.0))
        orig_params = float(configs.get('original_ssl_num_params', 1.0))
        warmup_epochs = configs.get('sparsity_warmup_epochs', 5)
        sparsity_schedule = configs.get('sparsity_schedule', 'cosine')
        min_sparsity = configs.get('min_sparsity', 0.0)


    frontend_type = configs['dataset_args'].get('frontend', 'fbank')
    for i, batch in enumerate(dataloader):
        cur_iter = (epoch - 1) * epoch_iter + i
        scheduler.step(cur_iter)
        margin_scheduler.step(cur_iter)

        # Calculate current target sparsity for progressive pruning
        if use_pruning:
            warmup_iters = warmup_epochs * epoch_iter
            if cur_iter < warmup_iters:
                # Use progressive pruning strategy during warmup
                target_sp_cur = get_progressive_sparsity(
                    current_iter=cur_iter,
                    total_warmup_iters=warmup_iters,
                    target_sparsity=target_sp,
                    schedule_type=sparsity_schedule,
                    min_sparsity=min_sparsity
                )
            else:
                # Use final target sparsity after warmup
                target_sp_cur = target_sp 

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
                outputs, cls_loss = outputs
            else:
                cls_loss = criterion(outputs, targets)
        
        # Calculate total loss with optional pruning regularization
        if use_pruning:
            cur_params = model.module.frontend.get_num_params()
            prune_loss, exp_sp = pruning_loss(
                cur_params, orig_params, target_sp_cur, l1, l2
            )            
            total_loss = cls_loss + prune_loss  
        else:
            prune_loss, exp_sp = 0.0, None
            total_loss = cls_loss

        # loss, acc
        loss_meter.add(total_loss.item())
        acc_meter.add(outputs.cpu().detach().numpy(), targets.cpu().numpy())

        # Update loss meters
        if use_pruning:
            cls_loss_meter.add(cls_loss.item())
            pruning_loss_meter.add(prune_loss.item())

        # Update model parameters
        optimizer.zero_grad()
        if use_pruning:
            optimizer_reg.zero_grad()
        
        # Backward pass with gradient scaling
        scaler.scale(total_loss).backward()
        
        # Update optimizers
        if use_pruning:
            scaler.step(optimizer_reg)
            with torch.no_grad():
                l1.clamp_(min=0.0)  # Ensure lambda1 >= 0
        scaler.step(optimizer)
        scaler.update()

        if (wandb_log):
            wandb_log.log({
                "learning_rate": scheduler.get_lr(),
                "train/loss": loss_meter.value()[0],
                "train/acc": acc_meter.value()[0]
            })
        # Log training progress
        if (i + 1) % configs['log_batch_interval'] == 0:
            row = [epoch, i + 1, scheduler.get_lr(),
                   margin_scheduler.get_margin(),
                   round(loss_meter.value()[0], 4),
                   round(acc_meter.value()[0], 2)]
            
            if use_pruning:
                row += [
                    round(cls_loss_meter.value()[0], 4),
                    round(pruning_loss_meter.value()[0], 4),
                    f"{target_sp_cur:.4f}",
                    f"{exp_sp:.4f}"
                ]
            
            logger.info(tp.row(tuple(row), width=10, style='grid'))

        if (i + 1) == epoch_iter:
            break

    # Final log for this epoch
    summary = [epoch, i + 1,
               scheduler.get_lr(),
               margin_scheduler.get_margin(),
               round(loss_meter.value()[0], 4),
               round(acc_meter.value()[0], 2)]
    
    if use_pruning:
        summary += [
            round(cls_loss_meter.value()[0], 4),
            round(pruning_loss_meter.value()[0], 4),
            f"{target_sp_cur:.4f}",
            f"{exp_sp:.4f}"
        ]    
    
    logger.info(tp.row(tuple(summary), width=10, style='grid'))
    


def val_epoch(val_dataloader,
              val_iter,
              model,
              criterion,
              device,
              configs,
              wandb_log=None):
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
                        features,
                        **configs['dataset_args'].get('cmvn_args', {}))

                outputs = model(features)  # (embed_a,embed_b) in most cases
                embeds = outputs[-1] if isinstance(outputs, tuple) else outputs
                outputs = model.module.projection(embeds, targets)
                if isinstance(outputs, tuple):
                    outputs, loss = outputs
                else:
                    loss = criterion(outputs, targets)

            # loss, acc
            val_loss_meter.add(loss.item())
            val_acc_meter.add(outputs.cpu().detach().numpy(),
                              targets.cpu().numpy())
            if (wandb_log):
                wandb_log.log({
                    "val/loss": val_loss_meter.value()[0],
                    "val/acc": val_acc_meter.value()[0]
                })

            if (i + 1) == val_iter:
                break

    return val_loss_meter.value()[0], val_acc_meter.value()[0]

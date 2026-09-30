import time
import torch
from tqdm import tqdm
from .utils import AverageMeter
from torch.amp import autocast
import torch.nn.functional as F
import torchvision.transforms.functional as fv
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision import transforms
import cv2
import os

def reshape_transform(tensor, height=16, width=16):
    result = tensor[:, 8:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def train(train_config, model, epoch, dataloader, loss_function, optimizer, scheduler=None, scaler=None):
    # # model.train()
    # model.eval()
    losses = AverageMeter()
    cafr_losses = AverageMeter()
    # wait before starting progress bar
    time.sleep(0.1)

    # Zero gradients for first step
    optimizer.zero_grad(set_to_none=True)

    step = 1


    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader

    for query, reference, ids, positions, mask,ground_name in bar:

        if scaler:
            # with torch.no_grad(), autocast():
            with autocast('cuda'):

                # data (batches) to device
                query = query.to(train_config.device)
                reference = reference.to(train_config.device)
                if len(positions):
                    positions = torch.cat((positions[0].unsqueeze(0), positions[1].unsqueeze(0)), 0).permute(1, 0).to(
                        train_config.device)

                # Forward pass
                features1, features2, cafr_loss, weights, _ = model(query, reference, positions, mask)
                if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1:
                    loss1 = loss_function(features1, features2, model.module.logit_scale.exp())

                else:
                    loss1 = loss_function(features1, features2, model.logit_scale.exp())
                if epoch > 3:
                    loss = loss1 + 0.05 * cafr_loss
                else:
                    loss = loss1

                current_loss = loss.item()
                cafr_loss_val = cafr_loss.item() if hasattr(cafr_loss, 'item') else cafr_loss
                losses.update(loss.item())
                cafr_losses.update(cafr_loss.item())

            scaler.scale(loss).backward()

            # Gradient clipping
            if train_config.clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)

                # Update model parameters (weights)
            scaler.step(optimizer)
            scaler.update()

            # Zero gradients for next step
            optimizer.zero_grad()

            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler == "constant":
                scheduler.step()

        else:

            # data (batches) to device
            query = query.to(train_config.device)
            reference = reference.to(train_config.device)

            # Forward pass
            features1, features2 = model(query, reference)
            if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1:
                loss = loss_function(features1, features2, model.module.logit_scale.exp())
            else:
                loss = loss_function(features1, features2, model.logit_scale.exp())
            losses.update(loss.item())

            # Calculate gradient using backward pass
            loss.backward()

            # Gradient clipping
            if train_config.clip_grad:
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)

                # Update model parameters (weights)
            optimizer.step()
            # Zero gradients for next step
            optimizer.zero_grad()

            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler == "constant":
                scheduler.step()

        if train_config.verbose:
            monitor = {"loss": "{:.4f}".format(loss.item()),
                       "feat_loss": "{:.4f}".format(loss1.item()),
                       "cafr_loss": "{:.4f}".format(cafr_loss.item()),
                       "loss_avg": "{:.4f}".format(losses.avg),
                       "posloss_avg": "{:.4f}".format(cafr_losses.avg),
                       "lr": "{:.6f}".format(optimizer.param_groups[0]['lr'])}

            bar.set_postfix(ordered_dict=monitor)
        del features1, features2, cafr_loss
        step += 1

    if train_config.verbose:
        bar.close()

    return losses.avg, cafr_losses.avg


def predict(train_config, model, dataloader):
    model.eval()
    # Get output shape from a dummy input
    dummy_input = torch.randn(1, *dataloader.dataset[0][0].shape, device=train_config.device)
    output_shape = model(dummy_input)[1].shape[1:]

    # Pre-allocate memory for efficiency (assuming fixed batch size)
    glob_feats = torch.zeros((len(dataloader.dataset), *output_shape), dtype=torch.float32, device=train_config.device)
    ids = torch.zeros(len(dataloader.dataset), dtype=torch.long, device=train_config.device)

    with torch.no_grad(), autocast('cuda'):
        for i, (img, ids_current) in enumerate(tqdm(dataloader)):
            img = img.to(train_config.device)
            _, glob_feat = model(img)

            # normalize is calculated in fp32
            if train_config.normalize_features:
                glob_feat = F.normalize(glob_feat, dim=-1)

            glob_feats[i * dataloader.batch_size:(i + 1) * dataloader.batch_size] = glob_feat
            ids[i * dataloader.batch_size:(i + 1) * dataloader.batch_size] = ids_current

    return glob_feats, ids



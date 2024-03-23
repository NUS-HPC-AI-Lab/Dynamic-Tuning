import os
import numpy as np
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from util.metrics import mean_per_class_accuracy, accuracy

import misc as misc
import util.lr_sched as lr_sched
from block_flops_dict import batch_select_flops
from misc import is_dist_avail_and_initialized
def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None, logger=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ", logger=logger)
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        logger.info('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples, targets = batch[0], batch[1]

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs, token_select = model(samples)
            outputs = dict(prediction=outputs, **token_select)
            loss, loss_dict = criterion(outputs, targets)

        loss_value = loss.item()
        loss_dict = {k: v.item() for k, v in loss_dict.items()}

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value, **loss_dict)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(loss=loss, lr=max_lr)

            
            
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_video_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None, logger=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ", logger=logger)
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        logger.info('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples, targets = batch[0], batch[1]

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs, token_select = model(samples)
            outputs = dict(prediction=outputs, **token_select)
            loss, loss_dict = criterion(outputs, targets)

        loss_value = loss.item()
        loss_dict = {k: v.item() for k, v in loss_dict.items()}

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value, **loss_dict)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(loss=loss, lr=max_lr)

            
            
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}




@torch.no_grad()
def evaluate(data_loader, model, device, logger, base_flops, flops_dict, args):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ", logger=logger)
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    token_select = []
    targets = []
    predictions = []
    
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[1]  # TODO: check why default use -1
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output, _ = model(images)
            loss = criterion(output, target)
            token_select.append(_["token_select"])
            predictions.append(output)
            targets.append(target)
            
        # acc1, acc5 = accuracy(output, target, topk=(1, 5))

        # batch_size = images.shape[0]
        # metric_logger.update(loss=loss.item())
        # metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        # metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        
    targets = torch.cat(targets, dim=0) # targets.shape 6149
    predictions = torch.cat(predictions, dim=0)
    token_select = torch.cat(token_select, dim=0)
    if is_dist_avail_and_initialized():
        targets = all_gather_concat(targets)
        predictions = all_gather_concat(predictions)
        token_select = all_gather_concat(token_select)
    print(token_select.shape)
    
    status = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    if args.metric == "accuracy":
        acc1, acc5 = accuracy(predictions, targets, topk=(1, 5))
        logger.info('* Acc@1 {:.3f} Acc@5 {:.3f}'.format(acc1, acc5))
        status["metric"] = acc1.item()
        
    elif args.metric == "mean_per_class_acc":
        class_mean_acc = mean_per_class_accuracy(predictions, targets, args.nb_classes)
        logger.info("mean per class accuracy={:4f}%".format(class_mean_acc))
        status["metric"] = class_mean_acc.item()
        
        
    token_select = token_select.float()
    assert "BASE" in args.finetune # block_num=12 only for ViT-b
    batch_flops = batch_select_flops(token_select.shape[0], flops_dict=flops_dict, token_select=token_select, block_num=12, base_flops=base_flops)
    logger.info("Average flops: {} GFlops".format(batch_flops.mean()))
    logger.info("Rate=flops/vit-b flops: {}".format(batch_flops.mean() / 17.6)) # vit-b
    
    
    
    logger.info("Select rate in different layers:")
    for layer in range(0, token_select.shape[1]): # [n, 12, 196, 1]
        logger.info(" {}% tokens selected in layer {}".format(token_select[:, layer, :, :].mean(), layer))
    logger.info("{}% tokens selected overall".format(token_select.mean()))
        
        

    return status

@torch.no_grad()
def evaluate_video(data_loader, model, device, logger, base_flops, flops_dict, args):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ", logger=logger)
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    token_select = []
    targets = []
    predictions = []
    
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[1]  # TODO: check why default use -1
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            B, V = images.shape[0], images.shape[1]
            images = images.flatten(0, 1)
            output, _ = model(images)
            output = output.view(B, V, -1).mean(dim=1)
            
            loss = criterion(output, target)
            token_select.append(_["token_select"])
            predictions.append(output)
            targets.append(target)
            
        # acc1, acc5 = accuracy(output, target, topk=(1, 5))

        # batch_size = images.shape[0]
        # metric_logger.update(loss=loss.item())
        # metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        # metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        
    targets = torch.cat(targets, dim=0) # targets.shape 6149
    predictions = torch.cat(predictions, dim=0)
    token_select = torch.cat(token_select, dim=0)
    if is_dist_avail_and_initialized():
        targets = all_gather_concat(targets)
        predictions = all_gather_concat(predictions)
        token_select = all_gather_concat(token_select)
    print(token_select.shape)
    
    status = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    if args.metric == "accuracy":
        acc1, acc5 = accuracy(predictions, targets, topk=(1, 5))
        logger.info('* Acc@1 {:.3f} Acc@5 {:.3f}'.format(acc1, acc5))
        status["metric"] = acc1.item()
        
    elif args.metric == "mean_per_class_acc":
        class_mean_acc = mean_per_class_accuracy(predictions, targets, args.nb_classes)
        logger.info("mean per class accuracy={:4f}%".format(class_mean_acc))
        status["metric"] = class_mean_acc.item()
        
        
    token_select = token_select.float()
    assert "BASE" in args.finetune # block_num=12 only for ViT-b
    batch_flops = batch_select_flops(token_select.shape[0], flops_dict=flops_dict, token_select=token_select, block_num=12, base_flops=base_flops)
    logger.info("Average flops: {} GFlops".format(batch_flops.mean()))
    logger.info("Rate=flops/vit-b flops: {}".format(batch_flops.mean() / 17.6)) # vit-b
    
    
    
    logger.info("Select rate in different layers:")
    for layer in range(0, token_select.shape[1]): # [n, 12, 196, 1]
        logger.info(" {}% tokens selected in layer {}".format(token_select[:, layer, :, :].mean(), layer))
    logger.info("{}% tokens selected overall".format(token_select.mean()))
        
        

    return status



def merge(eval_path, num_tasks, is_hmdb=False):
    dict_feats = {}
    dict_label = {}
    dict_pos = {}
    print("Reading individual output files")

    for x in range(num_tasks):
        file = os.path.join(eval_path, str(x) + '.txt')
        lines = open(file, 'r').readlines()[1:]
        for line in lines:
            line = line.strip()
            name = line.split('[')[0]
            label = line.split(']')[1].split(' ')[1]
            chunk_nb = line.split(']')[1].split(' ')[2]
            split_nb = line.split(']')[1].split(' ')[3]
            data = np.fromstring(line.split('[')[1].split(']')[0], dtype=np.float, sep=',')
            if not name in dict_feats:
                dict_feats[name] = []
                dict_label[name] = 0
                dict_pos[name] = []
            if chunk_nb + split_nb in dict_pos[name]:
                continue
            dict_feats[name].append(data)
            dict_pos[name].append(chunk_nb + split_nb)
            dict_label[name] = label
    print("Computing final results")

    input_lst = []
    print(len(dict_feats))
    for i, item in enumerate(dict_feats):
        input_lst.append([i, item, dict_feats[item], dict_label[item]])
    from multiprocessing import Pool
    p = Pool(64)
    ans = p.map(compute_video_hmdb if is_hmdb else compute_video, input_lst)
    top1 = [x[1] for x in ans]
    top5 = [x[2] for x in ans]
    pred = [x[0] for x in ans]
    label = [x[3] for x in ans]
    final_top1 ,final_top5 = np.mean(top1), np.mean(top5)
    return final_top1*100 ,final_top5*100


def compute_video(lst):
    i, video_id, data, label = lst
    feat = [x for x in data]
    feat = np.mean(feat, axis=0)
    pred = np.argmax(feat)
    top1 = (int(pred) == int(label)) * 1.0
    top5 = (int(label) in np.argsort(-feat)[:5]) * 1.0
    return [pred, top1, top5, int(label)]


def compute_video_hmdb(lst):
    i, video_id, data, label = lst
    feat = [x for x in data]
    feat = np.mean(feat, axis=0)
    # print(feat.shape)
    try:
        pred = np.argmax(feat)
        top1 = (int(pred) == int(label)) * 1.0
        top5 = (int(label) in np.argsort(-feat)[:5]) * 1.0
    except:
        pred = 0
        top1 = 1.0
        top5 = 1.0
        label = 0
    return [pred, top1, top5, int(label)]


def all_gather(data):

    world_size = misc.get_world_size()
    if world_size == 1:
        return [data]

    gather_list = [
        torch.empty_like(data)
        for _ in range(world_size)
    ]

    torch.distributed.all_gather(gather_list, data)

    return gather_list



def all_gather_concat(data: torch.Tensor) -> torch.Tensor:
    """Gather tensors with different first-dimension size and concat to one
    tenosr.

    Note:
        Only the first dimension should be different.

    Args:
        data (Tensor): Tensor to be gathered.

    Returns:
        torch.Tensor: The concatenated tenosr.
    """
    if misc.get_world_size() == 1:
        return data

    data_size = torch.tensor(data.size(0), device=data.device)

    sizes_list = all_gather(data_size)

    max_length = max(sizes_list)
    size_diff = max_length.item() - data_size.item()
    if size_diff:
        padding = torch.zeros(
            size_diff, *data.size()[1:], device=data.device, dtype=data.dtype)
        data = torch.cat((data, padding))

    gather_list = all_gather(data)

    all_data = []
    for tensor, size in zip(gather_list, sizes_list):

        all_data.append(tensor[:size])

    return torch.cat(all_data)


@torch.no_grad()
def final_test(data_loader, model, device, file):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Final_Test:'

    # switch to evaluation mode
    model.eval()
    final_result = []

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[1]
        ids = batch[2]
        chunk_nb = batch[3]
        split_nb = batch[4]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        for i in range(output.size(0)):
            string = "{} {} {} {} {}\n".format(
                ids[i], str(output.data[i].cpu().numpy().tolist()), str(int(target[i].cpu().numpy())),
                str(int(chunk_nb[i].cpu().numpy())), str(int(split_nb[i].cpu().numpy()))
            )
            final_result.append(string)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    if not os.path.exists(file):
        os.mknod(file)
    with open(file, 'w') as f:
        f.write("{}, {}\n".format(acc1, acc5))
        for line in final_result:
            f.write(line)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
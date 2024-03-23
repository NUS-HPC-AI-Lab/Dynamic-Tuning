import torch
from torch.nn import functional as F

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


def mean_per_class_accuracy(pred, target, num_classes):
    pred_score, pred_label = torch.topk(pred, k=1)
    pred_label = pred_label.flatten()
    
    pred_label = F.one_hot(pred_label, num_classes)
    target_label = F.one_hot(target, num_classes)
    class_correct = (pred_label & target_label)
    
    tp_sum = class_correct.sum(0)
    gt_sum = target_label.sum(0)
    recall = tp_sum / torch.clamp(gt_sum, min=1).float() * 100
    recall = recall.mean(0)
    return recall
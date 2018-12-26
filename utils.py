import torch
import torch.nn.init as init
import torch.nn as nn

def to_device(data, device):
    if isinstance(data, (list, tuple)):
        data = [to_device(d, device) for d in data]
    elif isinstance(data, torch.Tensor):
        data = data.to(device)
    else:
        raise TypeError("Unknown argument of type: {}".format(type(data)))
    return data

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class AverageListMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = []
        self.avg = []
        self.sum = []
        self.count = 0

    def update(self, val_list, n=1):
        if not isinstance(val_list, (tuple, list)):
            val_list = [val_list]
        if self.count == 0:
            size = len(val_list)
            self.val = [0] * size
            self.avg = [0] * size
            self.sum = [0] * size
        self.val = val_list
        self.sum = [(sum_i + val) for val, sum_i in zip(val_list, self.sum)]
        self.count += n
        self.avg = [(sum_i / self.count)for sum_i in self.sum]


def weight_init(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)
            if not m.bias is None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()
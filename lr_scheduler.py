from math import inf

class Step_decay:
    def __init__(self, decay_steps, gamma=0.1):
        self.base_lr = None
        self.decay_steps = list(decay_steps) + [inf]
        self.gamma = gamma
        self.type = type

    def __call__(self, optimizer, batch=None, epoch=None):
        if self.base_lr is None:
            self.base_lr = optimizer.param_groups[0]['lr']
        if batch is not None:
            step = next(i for i, x in enumerate(self.decay_steps) if x > batch)
        else:
            step = next(i for i, x in enumerate(self.decay_steps) if x > epoch)
        lr = self.base_lr * (self.gamma ** step)
        optimizer.param_groups[0]['lr'] = lr
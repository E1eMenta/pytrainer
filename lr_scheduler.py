from math import inf, cos, pi

class Step_decay:
    def __init__(self, decay_steps, gamma=0.1, T_warmup=0, warmup_lr=0):
        self.base_lr = None
        self.decay_steps = list(decay_steps) + [inf]
        self.gamma = gamma
        self.T_warmup = T_warmup
        self.warmup_lr = warmup_lr

    def __call__(self, optimizer, batch=None, epoch=None):
        if self.base_lr is None:
            self.base_lr = optimizer.param_groups[0]['lr']

        if batch is not None:
            iteration =  batch
        elif epoch is not None:
            iteration = epoch

        if iteration < self.T_warmup:
            lr = self.warmup_lr
        else:
            step = next(i for i, x in enumerate(self.decay_steps) if x > iteration)
            lr = self.base_lr * (self.gamma ** step)
        optimizer.param_groups[0]['lr'] = lr

class CosineAnnealingLR:
    def __init__(self, T_max, eta_min=0, T_warmup=0, warmup_lr=0):
        self.T_max = T_max
        self.eta_min = eta_min
        self.T_warmup = T_warmup
        self.warmup_lr = warmup_lr

        self.eta_max = None

    def __call__(self, optimizer, batch=None, epoch=None):
        if self.eta_max is None:
            self.eta_max = optimizer.param_groups[0]['lr']

        if batch is not None:
            iteration =  batch
        elif epoch is not None:
            iteration = epoch
        else:
            raise Exception("Unknown learning rate parameter")

        if iteration < self.T_warmup:
            lr = self.warmup_lr
        else:
            lr = self.eta_min + 1/2 * (self.eta_max - self.eta_min) * (1 + cos(iteration / self.T_max * pi))

        optimizer.param_groups[0]['lr'] = lr
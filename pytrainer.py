import sys
import torch

from .utils import to_device

from .callbacks import CallbackList, BaseLogger, CheckpointSaver, GitSaver, ValidationCallback

class Trainer:

    def compile(
            self,
            model,
            optimizer,
            criterion,
            callbacks=None,
            resume=None,
            validation=None,
            git_save=False
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.validation = validation
        self.git_save = git_save
        self.extra_callbacks = [callback for callback in callbacks if callback is not None]

        if resume:
            checkpoint = torch.load(resume)
            self.model.load_state_dict(checkpoint['model'].state_dict())

        self.epoch = 0
        self.iteration = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True

        self.model.to(self.device)
        self.model = torch.nn.DataParallel(self.model)
        self.model.train()



    def fit(
            self,
            dataloader,
            report_steps=100,
            val_steps=500,
            save_steps=5000,
            max_iter=10e20,
            max_epoch=10e15,
            tag=""
    ):
        # Set default self.callbacks
        self.callbacks = [
            BaseLogger(tag, report_steps),
            CheckpointSaver(save_steps)
        ]
        self.callbacks = CallbackList(self.callbacks)

        if self.git_save:
            self.callbacks.append(GitSaver())
        if self.validation:
            self.callbacks.append(ValidationCallback(self.validation, val_steps))
        [self.callbacks.append(extra_callback) for extra_callback in self.extra_callbacks]
        self.callbacks.set_params(self, self.model)

        self.logs = {
            "optimizer": self.optimizer,
            "save_fn": self.save_checkpoint
        }

        self.callbacks.on_train_begin(self.logs)
        try:
            while self.epoch < max_epoch:
                self.callbacks.on_epoch_begin(self.epoch, self.logs)

                for data in dataloader:
                    self.callbacks.on_batch_begin(self.iteration, self.logs)

                    data = to_device(data, self.device)
                    inputs, target = data

                    self.optimizer.zero_grad()
                    output = self.model(inputs)

                    loss = self.criterion(output, target)
                    self.logs["loss"] = loss
                    if isinstance(loss, torch.Tensor):
                        loss.backward()
                    elif isinstance(loss, (list, tuple)):
                        loss[0].backward()
                    elif isinstance(loss, dict):
                        loss["total"].backward()

                    self.optimizer.step()

                    self.iteration += 1
                    if self.iteration > max_iter:
                        self.finish_train()

                    self.callbacks.on_batch_end(self.iteration, self.logs)

                self.epoch += 1
                self.callbacks.on_epoch_end(self.epoch, self.logs)
        except KeyboardInterrupt:
            self.finish_train()

        self.callbacks.on_train_end(self.logs)
        return self.model

    def finish_train(self):
          print("Train of model {} finished".format(self.model.module.__class__.__name__))
          self.callbacks.on_train_end(self.logs)
          sys.exit(0)

    def save_checkpoint(self, filename):
        model = self.model.module
        torch.save({
            'model':     model,
            'iteration': self.iteration,
            'optimizer': self.optimizer.state_dict(),
            'epoch':    self.epoch
        }, filename)
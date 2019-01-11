import os
import time
import torch
import datetime
from subprocess import call, DEVNULL

from tensorboardX import SummaryWriter

from .utils import AverageMeter, AverageListMeter


class Callback:

    def set_params(self, trainer, model):
        self.trainer = trainer
        self.model = model.module

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

class CallbackList(object):
    """Container abstracting a list of callbacks.
    # Arguments
        callbacks: List of `Callback` instances.
        queue_length: Queue length for keeping
            running statistics over callback execution time.
    """

    def __init__(self, callbacks=None):
        callbacks = callbacks or []
        self.callbacks = [c for c in callbacks]

    def append(self, callback):
        self.callbacks.append(callback)

    def set_params(self, trainer, model):
        for callback in self.callbacks:
            callback.set_params(trainer, model)

    def set_model(self, model):
        for callback in self.callbacks:
            callback.set_model(model)

    def on_epoch_begin(self, epoch, logs=None):
        """Called at the start of an epoch.
        # Arguments
            epoch: integer, index of epoch.
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of an epoch.
        # Arguments
            epoch: integer, index of epoch.
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch, logs=None):
        """Called right before processing a batch.
        # Arguments
            batch: integer, index of batch within the current epoch.
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs=None):
        """Called at the end of a batch.
        # Arguments
            batch: integer, index of batch within the current epoch.
            logs: dictionary of logs.
        """
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)

    def on_train_begin(self, logs=None):
        """Called at the beginning of training.
        # Arguments
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        """Called at the end of training.
        # Arguments
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def __iter__(self):
        return iter(self.callbacks)


class BaseLogger(Callback):

    def __init__(self, tag, report_steps):
        self.tag = tag
        self.report_steps = report_steps

    def on_train_begin(self, logs=None):
        logs = logs or {}

        model_name = self.model.__class__.__name__
        date = datetime.datetime.now()
        date = "{}-{}-{}_{}-{}".format(date.day, date.month, date.year, date.hour, date.minute)
        base_dir = model_name + "_" + self.tag
        logdir = os.path.join(base_dir, date)


        self.summary_dir = os.path.join(logdir, "summary")
        self.save_dir = os.path.join(logdir, "saved")
        os.makedirs(self.summary_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)

        self.writer = SummaryWriter(self.summary_dir)

        logs["base_dir"] = base_dir
        logs["date"] = date
        logs["logdir"] = logdir
        logs["summary_dir"] = self.summary_dir
        logs["save_dir"] = self.save_dir
        logs["writer"] = self.writer

        print("Logging folder: {}".format(logdir))

        self.loss_accumulator = AverageListMeter()
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()

        self.start_time = None
        self.end_time = None

        self.loss_names = None

    def on_epoch_begin(self, epoch, logs=None):
        logs["epoch"] = epoch

    def on_batch_begin(self, batch, logs=None):
        logs["batch"] = batch
        self.start_time = time.time()
        if self.end_time:
            self.data_time.update(self.start_time - self.end_time)

    def on_batch_end(self, batch, logs=None):
        losses = logs["loss"]
        if isinstance(losses, (dict)) and self.loss_names:
            self.loss_names = [loss_name for loss_name in losses]


        if isinstance(losses, (dict)):
            losses = [losses[name] for name in self.loss_names]
        self.loss_accumulator.update(losses)

        if batch > 0 and batch % self.report_steps == 0:
            print(
                "Iter: {0}(Epoch: {1}), Batch time {2:.4}, Data load time {3:.4} ".format(
                    batch,
                    logs["epoch"],
                    self.batch_time.avg,
                    self.data_time.avg
                ),
                end=" "
            )
            for idx, avg in enumerate(self.loss_accumulator.avg):
                loss_name = self.loss_names[idx] if self.loss_names else "Loss" + str(idx)
                print(loss_name + ": {0:.4}, ".format(avg.cpu().item()), end=" ")
                self.writer.add_scalar(loss_name, avg, batch)
            print()
            self.writer.add_scalar("lr", logs["optimizer"].param_groups[0]['lr'], batch)

            self.loss_accumulator.reset()
            self.batch_time.reset()
            self.data_time.reset()


        self.end_time = time.time()
        self.batch_time.update(self.end_time - self.start_time)

class CheckpointSaver(Callback):

    def __init__(self, save_steps):
        self.save_steps = save_steps


    def on_batch_end(self, batch, logs=None):
        if batch > 0 and batch % self.save_steps == 0:
            filename = logs["save_dir"] + "/iter_{0}.pth".format(logs["batch"])
            save_fn = logs["save_fn"]
            save_fn(filename)

class GitSaver(Callback):

    def on_train_begin(self, logs=None):
        if call(["git", "status"], stdout=DEVNULL, stderr=DEVNULL) != 0:
            call(["git", "init"], stdout=DEVNULL, stderr=DEVNULL)
            call(["git", "add", "*.py"], stdout=DEVNULL, stderr=DEVNULL)
            call(["git", "-c", "user.name='None'", "-c", "user.email='None'", "commit", "-m", 'Init commit'],
                 stdout=DEVNULL, stderr=DEVNULL)

        call(["git", "-c", "user.name='None'", "-c", "user.email='None'", "stash"],
             stdout=DEVNULL, stderr=DEVNULL)
        call(["git", "branch", "{}".format(logs["base_dir"])], stdout=DEVNULL, stderr=DEVNULL)
        call(["git", "checkout", "{}".format(logs["base_dir"])], stdout=DEVNULL, stderr=DEVNULL)
        call(["git", "stash", "apply"], stdout=DEVNULL, stderr=DEVNULL)
        call(["git", "add", "-u"], stdout=DEVNULL, stderr=DEVNULL)
        call(["git", "add", "*.py"], stdout=DEVNULL, stderr=DEVNULL)
        call(["git", "-c", "user.name='None'", "-c", "user.email='None'", "commit", "-m", '{}'.format(logs["date"])],
             stdout=DEVNULL, stderr=DEVNULL)
        call(["git", "checkout", "-"], stdout=DEVNULL, stderr=DEVNULL)
        call(["git", "stash", "pop"], stdout=DEVNULL, stderr=DEVNULL)

class LearningRateScheduler(Callback):

    def __init__(self, schedule, type="epoch"):
        self.schedule = schedule
        self.type = type

    def on_epoch_begin(self, epoch, logs=None):
        if self.type == "epoch":
            self.schedule(logs["optimizer"], epoch=epoch)

    def on_batch_begin(self, batch, logs=None):
        if self.type == "batch":
            self.schedule(logs["optimizer"], batch=batch)

class ValidationCallback(Callback):

    def __init__(self, validator, val_steps):
        self.validator = validator
        self.val_steps = val_steps

    def on_batch_end(self, batch, logs=None):
        if batch > 0 and batch % self.val_steps == 0:
            torch.cuda.empty_cache()
            self.model.eval()
            with torch.no_grad():
                self.validator(model=self.model, params=logs)
            self.model.train()
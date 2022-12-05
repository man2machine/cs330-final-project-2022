# -*- coding: utf-8 -*-
"""
Created on Sat May  8 15:05:21 2021

@author: Shahir
"""

import os
import time
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm

from cs330_project.utils import get_timestamp_str


class ModelTracker:
    def __init__(
            self,
            root_dir):

        experiment_dir = "Experiment {}".format(get_timestamp_str())
        self.save_dir = os.path.join(root_dir, experiment_dir)
        self.best_model_metric = float("-inf")
        self.record_per_epoch = {}

    def update_info_history(
            self,
            epoch,
            info):

        os.makedirs(self.save_dir, exist_ok=True)
        self.record_per_epoch[epoch] = info
        fname = "Experiment Epoch Info History.pckl"
        with open(os.path.join(self.save_dir, fname), "wb") as f:
            pickle.dump(self.record_per_epoch, f)

    def update_model_weights(
            self,
            epoch,
            model_state_dict,
            metric=None,
            save_best=True,
            save_latest=True,
            save_current=False):

        os.makedirs(self.save_dir, exist_ok=True)
        update_best = metric is None or metric > self.best_model_metric
        if update_best and metric is not None:
            self.best_model_metric = metric

        if save_best and update_best:
            torch.save(
                model_state_dict, os.path.join(
                    self.save_dir, "Weights Best.pckl")
            )
        if save_latest:
            torch.save(
                model_state_dict, os.path.join(
                    self.save_dir, "Weights Latest.pckl")
            )
        if save_current:
            torch.save(
                model_state_dict,
                os.path.join(
                    self.save_dir,
                    "Weights Epoch {} {}.pckl".format(
                        epoch, get_timestamp_str())
                )
            )


def make_optimizer(
        model,
        lr=0.001,
        weight_decay=0.0,
        clip_grad_norm=False,
        verbose=False):

    # Get all the parameters
    params_to_update = model.parameters()

    if verbose:
        print("Params to learn:")
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                print("\t", name)

    optimizer = optim.Adam(
        params_to_update,
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=weight_decay,
        amsgrad=True
    )

    if clip_grad_norm:
        nn.utils.clip_grad_norm_(params_to_update, 3.0)

    return optimizer


def get_lr(
        optimizer):

    for param_group in optimizer.param_groups:
        return param_group["lr"]


def set_optimizer_lr(
        optimizer,
        lr):

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def make_scheduler(
        optimizer):

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        threshold=0.025,
        patience=3,
        cooldown=3,
        min_lr=1e-6,
        verbose=True
    )
    return scheduler


def train_one_epoch(
        model,
        criterion,
        dataloader,
        optimizer,
        device,
        grad_accum_steps=1):

    model.train()

    optimizer.zero_grad()

    for data_iter_step, (samples, targets) in enumerate(dataloader):
        if len(samples.shape) == 6:
            b, r, c, t, h, w = samples.shape
            samples = samples.view(b * r, c, t, h, w)
            targets = targets.view(b * r)

        samples = samples.to(device)
        targets = targets.to(device)

        with torch.set_grad_enabled(True):
            outputs = model(samples)
            loss, _, _ = outputs
            # loss = criterion(outputs, targets)
            loss /= grad_accum_steps
        loss.backward()

        if (data_iter_step + 1) % grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        if device.type == 'cuda':
            torch.cuda.synchronize()


def train_model(
        device,
        model,
        dataloaders,
        criterion,
        optimizer,
        save_dir,
        lr_scheduler=None,
        save_model=False,
        save_best=False,
        save_latest=False,
        save_all=False,
        save_log=False,
        num_epochs=1):

    start_time = time.time()

    tracker = ModelTracker(save_dir)

    for epoch in range(num_epochs):
        print("-" * 10)
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print("-" * 10)

        train_loss_info = {}

        print("Training")
        
        train_one_epoch(
            model,
            criterion,
            dataloaders['train'],
            optimizer,
            device)

        print("Training Loss: {:.4f}".format(training_loss))
        print("Training Accuracy: {:.4f}".format(training_acc))
        train_loss_info["loss"] = train_loss_record

        print("Testing")
        # TODO
        
        print("Testing loss: {:.4f}".format(test_loss))
        print("Testing accuracy: {:.4f}".format(test_accuracy))

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if save_model:
            model_weights = model.state_dict()
            tracker.update_model_weights(
                epoch,
                model_weights,
                metric=test_accuracy,
                save_best=save_best,
                save_latest=save_latest,
                save_current=save_all
            )

        if save_log:
            info = {"train_loss_history": train_loss_info}
            tracker.update_info_history(epoch, info)

        if lr_scheduler:
            lr_scheduler.step()

        print()

    time_elapsed = time.time() - start_time
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )

    return tracker


def load_weights(
        model,
        weights_fname,
        map_location=None):

    model.load_state_dict(torch.load(weights_fname, map_location=map_location))
    return model


def save_training_session(
        model,
        optimizer,
        sessions_save_dir):

    sub_dir = "Session {}".format(get_timestamp_str())
    sessions_save_dir = os.path.join(sessions_save_dir, sub_dir)
    os.makedirs(sessions_save_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(
        sessions_save_dir, "Model State.pckl"))
    torch.save(
        optimizer.state_dict(), os.path.join(sessions_save_dir, "Optimizer State.pckl")
    )

    print("Saved session to", sessions_save_dir)


def load_training_session(
        model,
        optimizer,
        session_dir,
        update_models=True,
        map_location=None):

    if update_models:
        model.load_state_dict(
            torch.load(
                os.path.join(session_dir, "Model State.pckl"), map_location=map_location
            )
        )
        optimizer.load_state_dict(
            torch.load(
                os.path.join(session_dir, "Optimizer State.pckl"),
                map_location=map_location
            )
        )

    print("Loaded session from", session_dir)

    out_data = {"model": model, "optimizer": optimizer}

    return out_data

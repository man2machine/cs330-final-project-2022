# -*- coding: utf-8 -*-
"""
Created on Sat May  8 15:05:21 2021

@author: Shahir, Faraz, Pratyush
"""

import os
import time
import pickle

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm.auto import tqdm

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


def train_mae_single_epoch(
        model,
        criterion,
        dataloader,
        optimizer,
        device,
        grad_accum_steps=1):

    model.train()
    optimizer.zero_grad()

    loss_record = []
    running_loss = 0.0
    running_count = 0

    pbar = tqdm(dataloader)
    for data_iter_step, (samples, labels) in enumerate(pbar):
        visual_data, mask_info = samples
        visual_data = visual_data.to(device)
        mask_info = [n.to(device) for n in mask_info]

        with torch.set_grad_enabled(True):
            outputs = model(visual_data, mask_info)
            latent_patched, x_patched, x_hat_patched = outputs
            loss = criterion(x_hat_patched, x_patched)
            loss /= grad_accum_steps
        loss.backward()
        
        current_loss = loss.detach().item()
        num_samples = visual_data.size(0)
        
        running_loss += current_loss * num_samples
        running_count += num_samples
        
        avg_loss = running_loss / running_count
        
        loss_record.append(current_loss)

        desc = "Avg. Loss: {:.4f}, Current Loss: {:.4f}"
        desc = desc.format(avg_loss, current_loss)
        pbar.set_description(desc)

        if (data_iter_step + 1) % grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        if device.type == 'cuda':
            torch.cuda.synchronize()

    return loss_record, avg_loss


def train_mae_model(
        device,
        model,
        dataloader,
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

        train_loss_record, training_loss = train_mae_single_epoch(
            model,
            criterion,
            dataloader,
            optimizer,
            device)

        print("Training Loss: {:.4f}".format(training_loss))
        train_loss_info["loss"] = train_loss_record

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if save_model:
            model_weights = model.state_dict()
            tracker.update_model_weights(
                epoch,
                model_weights,
                metric=training_loss,
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
    print("Training complete in {:.0f}m {:.0f}s".format(
        time_elapsed // 60, time_elapsed % 60))

    return tracker


def train_classifier_single_epoch(
        model,
        criterion,
        dataloader,
        optimizer,
        device,
        grad_accum_steps=1):

    model.train()
    optimizer.zero_grad()

    loss_record = []
    running_loss = 0.0
    running_correct = 0
    running_count = 0

    pbar = tqdm(dataloader)
    for data_iter_step, (samples, labels) in enumerate(pbar):
        visual_data, mask_info = samples
        visual_data = visual_data.to(device)

        with torch.set_grad_enabled(True):
            outputs = model(visual_data)
            _, preds = torch.max(outputs, dim=1)
            loss = criterion(outputs, labels)
            loss /= grad_accum_steps
        loss.backward()
        
        correct = torch.sum(preds == labels).item()
        current_loss = loss.detach().item()
        num_samples = visual_data.size(0)
        
        running_correct += correct
        running_loss += current_loss * num_samples
        running_count += num_samples
        
        avg_loss = running_loss / running_count
        avg_acc = running_correct / running_count
        
        loss_record.append(current_loss)
        
        if (data_iter_step + 1) % grad_accum_steps == 0:
            desc = "Avg. Loss: {:.4f}, Current Loss: {:.4f}, Acc: {:4f}"
            desc = desc.format(avg_loss, current_loss, avg_acc)
            pbar.set_description(desc)

            optimizer.step()
            optimizer.zero_grad()

        if device.type == 'cuda':
            torch.cuda.synchronize()

    return loss_record, avg_loss, avg_acc


def test_classifier_single_epoch(
        model,
        criterion,
        dataloader,
        device):

    model.eval()
    
    running_loss = 0.0
    running_correct = 0
    running_count = 0

    pbar = tqdm(dataloader)
    for data_iter_step, (samples, labels) in enumerate(pbar):
        visual_data, mask_info = samples
        visual_data = visual_data.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(visual_data)
            _, preds = torch.max(outputs, dim=1)
            loss = criterion(outputs, labels)
        
        correct = torch.sum(preds == labels).item()
        current_loss = loss.detach().item()
        num_samples = visual_data.size(0)
        
        running_correct += correct
        running_loss += current_loss * num_samples
        running_count += num_samples
        
        avg_loss = running_loss / running_count
        avg_acc = running_correct / running_count

        desc = "Avg. Loss: {:.4f}, Current Loss: {:.4f}, Acc: {:4f}"
        desc = desc.format(avg_loss, current_loss, avg_acc)
        pbar.set_description(desc)

        if device.type == 'cuda':
            torch.cuda.synchronize()

    return avg_loss, avg_acc


def train_classifier_model(
        device,
        model,
        dataloader_train,
        dataloader_test,
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

        train_info = {}
        test_info = {}

        print("Training")

        train_loss_record, training_loss, training_acc = train_classifier_single_epoch(
            model,
            criterion,
            dataloader_train,
            optimizer,
            device)

        print("Training Loss: {:.4f}".format(training_loss))
        train_info["loss_record"] = train_loss_record
        train_info["loss_avg"] = training_loss
        train_info["acc_avg"] = training_acc
        
        print("Testing")

        testing_loss, testing_acc = test_classifier_single_epoch(
            model,
            criterion,
            dataloader_test,
            optimizer,
            device)

        print("Testing Loss: {:.4f}, Testing Acc.: {:.4f}".format(testing_loss, testing_acc))
        test_info["loss_avg"] = testing_loss
        test_info["acc_avg"] = testing_acc

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if save_model:
            model_weights = model.state_dict()
            tracker.update_model_weights(
                epoch,
                model_weights,
                metric=training_loss,
                save_best=save_best,
                save_latest=save_latest,
                save_current=save_all
            )

        if save_log:
            info = {"train_loss_history": train_info}
            tracker.update_info_history(epoch, info)

        if lr_scheduler:
            lr_scheduler.step()

        print()

    time_elapsed = time.time() - start_time
    print("Training complete in {:.0f}m {:.0f}s".format(
        time_elapsed // 60, time_elapsed % 60))

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

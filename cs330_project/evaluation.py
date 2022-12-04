# -*- coding: utf-8 -*-
"""
Created on Sat May  8 15:05:32 2021

@author: Shahir
"""

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import torch

def get_dataloader_stats(dataloader, model, criterion, device, max_iter=None):
    with torch.set_grad_enabled(False):
        model.eval()
        
        input_datas = []
        label_datas = []
        output_datas = []
        pred_datas = []
        
        running_loss = 0.0
        running_correct = 0
        running_count = 0
        n = 0

        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, dim=1)
            correct_count = (preds == labels).sum()

            input_datas.append(inputs.detach().cpu().numpy())
            label_datas.append(labels.detach().cpu().numpy())
            output_datas.append(outputs.detach().cpu().numpy())
            pred_datas.append(preds.detach().cpu().numpy())
            
            running_loss += loss.detach().item() * inputs.size(0)
            running_correct += correct_count.detach().item()
            running_count += inputs.size(0)
            avg_loss = running_loss / running_count
            avg_acc = running_correct / running_count

            n += 1
            if max_iter and n == max_iter:
                break
            
    stats = {
        "inputs": np.concatenate(input_datas),
        "labels": np.concatenate(label_datas),
        "outputs": np.concatenate(output_datas),
        "preds": np.concatenate(pred_datas),
        "loss": avg_loss,
        "acc": avg_acc
    }
    
    return stats


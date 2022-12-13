# based onhttps://github.com/aanna0701/SPT_LSA_ViT
#@author:  Faraz, Shahir, Pratyush
import argparse

import os
import random
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor
#from load_data import DataGenerator
from torch.utils.tensorboard import SummaryWriter
import torchvision
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
#change this to cuda on azure
DEVICE = torch.device("cpu")
print(DEVICE)

def initialize_weights(model):
    if type(model) in [nn.Linear]:
        nn.init.xavier_uniform_(model.weight)
        nn.init.zeros_(model.bias)
    elif type(model) in [nn.LSTM, nn.RNN, nn.GRU]:
        nn.init.orthogonal_(model.weight_hh_l0)
        nn.init.xavier_uniform_(model.weight_ih_l0)
        nn.init.zeros_(model.bias_hh_l0)
        nn.init.zeros_(model.bias_ih_l0)


class VitMaskedEncoderWithMetaHead(nn.Module):
    def __init__(self, num_classes, vitmaskedvideoautoencoder, samples_per_class, hidden_dim,
                 num_layers=1, rnn_type="lstm"):
        super(VitMaskedEncoderWithMetaHead, self).__init__()
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        self.hidden_dim = 128

        #we are setting it from calling class
        self.vitmaskedvideoautoencoder=vitmaskedvideoautoencoder
        self.latentOriginalprojection =vitmaskedvideoautoencoder.num_patches * vitmaskedvideoautoencoder.dim

        kernel_size=3
        channels_out= 50
        #self.reducedim = nn.Linear(self.latentOriginalprojection , self.latentReduceprojection, bias=False)
        self.reducedim =  nn.Conv1d(vitmaskedvideoautoencoder.num_patches, channels_out, kernel_size)
        L = vitmaskedvideoautoencoder.dim - (kernel_size-1)
        self.latentReduceprojection =  channels_out * L

        if rnn_type == "lstm":
            self.layer1 = torch.nn.LSTM(self.latentReduceprojection +  num_classes , hidden_dim, batch_first=True,
                                        num_layers=num_layers)
            self.layer2 = torch.nn.LSTM(hidden_dim, num_classes, batch_first=True)
        elif rnn_type == "gru":
            self.layer1 = torch.nn.GRU(self.latentReduceprojection +   num_classes, hidden_dim, batch_first=True,
                                       num_layers=num_layers)
            self.layer2 = torch.nn.GRU(hidden_dim, num_classes, batch_first=True)
        elif rnn_type == "rnn":
            self.layer1 = torch.nn.RNN(self.latentReduceprojection +    num_classes, hidden_dim, batch_first=True,
                                       num_layers=num_layers)
            self.layer2 = torch.nn.RNN(hidden_dim, num_classes, batch_first=True)
        else:
            raise ValueError()
        initialize_weights(self.layer1)
        initialize_weights(self.layer2)

    def forward(self, input_images, input_labels):
        """
        MANN
        Args:
            input_images: [B, K+1, C,D, W, H] flattened images
            labels: [B, K+1, N, 1 ] ground truth labels
        Returns:
            [B, K+1, N, N] predictions
        """
        #############################
        #### YOUR CODE GOES HERE ####


        #k=k+1 (number of shots(support) plus 1(query)
        # learn latent representation from encoder
        B= input_images.shape[0]  #Batch size
        K =input_images.shape[1] #KShot
        N = input_images.shape[2] #N Way
        input_images =  rearrange(input_images, 'b  k n c d w h-> (b k n) c d w h')
        #t=token count
        print(self.latentOriginalprojection)
        latent_videos = self.vitmaskedvideoautoencoder.forward(input_images)
        latent_videos = self.reducedim(latent_videos)
        latent_videos =  rearrange(latent_videos, ' (b k n) t d -> b k n (t d)',b=B,k=K,n=N )
        #latent_videos = self.reducedim(latent_videos)
        input_labels = torch.nn.functional.one_hot(input_labels, self.vitmaskedvideoautoencoder.num_classes)


        """
               MANN
               Args:
                   input_images: [B, K+1, N, 784] flattened images
                   labels: [B, K+1, N, N] ground truth labels
               Returns:
                   [B, K+1, N, N] predictions
               """
        #############################
        #### YOUR CODE GOES HERE ####
        _, K, N, I = latent_videos.shape
        #############################
        #### YOUR CODE GOES HERE ####
        input_labels = torch.clone(input_labels)
        input_labels[:, -1] = 0
        inputs = torch.concat((latent_videos, input_labels), dim=-1).float()
        b, kp, n, d = inputs.shape
        inputs = inputs.view((b, -1, d))

        outputs, _ = self.layer1(inputs)
        outputs, _ = self.layer2(outputs)

        outputs = outputs.view((b, kp, n, n))

        return outputs

    def loss_function(self, preds, labels):
        """
        Computes MANN loss
        Args:
            preds: [B, K+1, N, N] network output
            labels: [B, K+1, N, N] labels
        Returns:
            scalar loss
        Note:
            Loss should only be calculated on the N test images
        """
        #############################

        #### YOUR CODE GOES HERE ####
        #do one hot encoding
        print(labels.shape[-1])
        labels = torch.nn.functional.one_hot(labels, labels.shape[-1] )
        return F.cross_entropy(F.softmax(preds[:, -1].to(torch.float32), dim=2), labels[:, -1].to(torch.float32))
        #############################


def train_step(images, labels, model, optim, eval=False):
    predictions = model(images, labels)
    loss = model.loss_function(predictions, labels)
    if not eval:
        optim.zero_grad()
        loss.backward()
        optim.step()
    return predictions.detach(), loss.detach()



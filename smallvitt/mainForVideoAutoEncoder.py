#from utils.mix import cutmix_data, mixup_data, mixup_criterion
import numpy as np
import random
import logging as log
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from colorama import Fore, Style
from torchsummary import summary
from utils.losses import LabelSmoothingCrossEntropy
import os
from utils.sampler import RASampler
# import models.create_model as m
from utils.logger_dict import Logger_dict
from utils.print_progress import progress_bar
from utils.training_functions import accuracy
import argparse
#from utils.scheduler import build_scheduler
from utils.dataloader import datainfo, dataload
from models.create_model import create_model
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import warnings
#import utils.loader as loader
from datasets import DataAugmentationForVideoMAE
from datasets import VideoMAE
from utils.scheduler import build_scheduler
warnings.filterwarnings("ignore", category=Warning)

best_acc1 = 0
MODELS = ['vitMaskedVideoAutoEncoder', 'vitMaskedVideoEncoderWithHead', 'vitMaskedVideoEncoderWithMetaHead' ]


def init_parser():
    parser = argparse.ArgumentParser(description='TinyVirat Dataset')
    # parser.add_argument('--model', default='vit', type=str, help='model vit')
    # Data args
    #parser.add_argument('--data_path', default='./TinyVIRAT', type=str, help='dataset path')

    parser.add_argument('--dataset', default='TinyVIRAT', choices=['TinyVIRAT'], type=str,
                        help='Image Net dataset path')

    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    parser.add_argument('--print-freq', default=1, type=int, metavar='N', help='log frequency (by iteration)')

    # Optimization hyperparams
    parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')

    parser.add_argument('--warmup', default=10, type=int, metavar='N', help='number of warmup epochs')

    parser.add_argument('-b', '--batch_size', default=4, type=int, metavar='N', help='mini-batch size (default: 128)',
                        dest='batch_size')

    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')

    parser.add_argument('--weight-decay', default=5e-2, type=float, help='weight decay (default: 1e-4)')

    parser.add_argument('--model', type=str, default='vitmaskedvideoencoderwithhead', choices=MODELS)

    parser.add_argument('--disable-cos', action='store_true', help='disable cosine lr schedule')

    parser.add_argument('--enable_aug', action='store_true', help='disable augmentation policies for training')

    parser.add_argument('--gpu', default=0, type=int)

    parser.add_argument('--no_cuda', action='store_true', help='disable cuda')

    parser.add_argument('--ls', action='store_false', help='label smoothing')

    parser.add_argument('--channel', type=int, help='disable cuda')

    parser.add_argument('--heads', type=int, help='disable cuda')

    parser.add_argument('--depth', type=int, help='disable cuda')

    parser.add_argument('--tag', type=str, help='tag', default='')

    parser.add_argument('--input_size', default=32, type=int,
                        help='videos input size for backbone')


    #parser.add_argument('--seed', type=int, default=0, help='seed')

    parser.add_argument('--sd', default=0.1, type=float, help='rate of stochastic depth')

    #parser.add_argument('--resume', default=False, help='Version')

    parser.add_argument('--aa', action='store_false', help='Auto augmentation used'),

    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')

    parser.add_argument('--cm', action='store_false', help='Use Cutmix')

    parser.add_argument('--beta', default=1.0, type=float,
                        help='hyperparameter beta (default: 1)')

    parser.add_argument('--mu', action='store_false', help='Use Mixup')

    parser.add_argument('--alpha', default=1.0, type=float,
                        help='mixup interpolation coefficient (default: 1)')

    parser.add_argument('--mix_prob', default=0.5, type=float,
                        help='mixup probability')

    parser.add_argument('--ra', type=int, default=3, help='repeated augmentation')

    parser.add_argument('--t', type=int, default=16, help="frame depth")

    parser.add_argument('--re', default=0.25, type=float, help='Random Erasing probability')

    parser.add_argument('--re_sh', default=0.4, type=float, help='max erasing area')

    parser.add_argument('--re_r1', default=0.3, type=float, help='aspect of erasing area')

    parser.add_argument('--is_LSA', action='store_true', help='Locality Self-Attention')

    parser.add_argument('--is_SPT', action='store_true', help='Shifted Patch Tokenization')

    # Dataset parameters
    parser.add_argument('--data_path', default='TinyVIRAT/tiny_train.json', type=str,
                        help='dataset path')
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--sampling_rate', type=int, default=4)
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem',
                        help='')

    parser.add_argument('--mask_type', default='tube', choices=['random', 'tube'],
                        type=str, help='masked strategy of video tokens/patches')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='ratio of the visual tokens/patches need be masked')

    parser.set_defaults(pin_mem=True)


    return parser

#change this to cuda on azure
DEVICE = torch.device("cpu")
print(DEVICE)

import sys
#"compositeDataset/train/"
def build_pretraining_dataset(args, videoPath, classToLabelMap=None):
    transform = DataAugmentationForVideoMAE(args)
    dataset = VideoMAE(
        root=videoPath,
        setting=args.data_path,
        video_ext='mp4',
        is_color=True,
        modality='rgb',
        new_length=args.num_frames,
        new_step=args.sampling_rate,
        transform=transform,
        labelMap=classToLabelMap,
        temporal_jitter=False,
        video_loader=True,
        use_decord=True,
        lazy_init=False

    )
    print("Data Aug = %s" % str(transform))
    return dataset, dataset.codingToLabel





# Logging and Visualization




def main(args):
    global best_acc1

    data_info = datainfo(logger, args)



    patch_size=4

    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.num_frames // 2, args.input_size // patch_size, args.input_size // patch_size)
    args.patch_size = patch_size

    # get dataset
    train_dataset,classToLabelTrain = build_pretraining_dataset(args, "compositeDataset/train/")
    val_dataset,classToLabelValid = build_pretraining_dataset(args, "compositeDataset/test/", classToLabelTrain)



    #test if we are getting same class to label set for both validation and training set
    for classNum in classToLabelValid:
        classLabelTrain = classToLabelTrain[classNum]
        classLabelValid = classToLabelValid[classNum]
        print(classLabelTrain)
        print(classLabelValid)
        if classLabelValid != classLabelTrain:
            print("error in mapping")
    #override image size with numofclasses we got from

    data_info['n_classes'] = len(classToLabelValid)
    model = create_model(data_info['img_size'], data_info['n_classes'], args)
    patch_size = model.patch_dim
    model.to(DEVICE)

    #    summary(model, input_size=(1, 1, 192))
    print(Fore.GREEN + '*' * 80)
    logger.debug(f"Creating model: {model_name}")
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.debug(f'Number of params: {format(n_parameters, ",")}')
    logger.debug(f'Initial learning rate: {args.lr:.6f}')
    logger.debug(f"Start training for {args.epochs} epochs")
    print('*' * 80 + Style.RESET_ALL)

    if args.ls:
        print(Fore.YELLOW + '*' * 80)
        logger.debug('label smoothing used')
        print('*' * 80 + Style.RESET_ALL)
        criterion = LabelSmoothingCrossEntropy()

    else:
        criterion = nn.CrossEntropyLoss()

    if args.sd > 0.:
        print(Fore.YELLOW + '*' * 80)
        logger.debug(f'Stochastic depth({args.sd}) used ')
        print('*' * 80 + Style.RESET_ALL)

    criterion = criterion.cuda(args.gpu)

    normalize = [transforms.Normalize(mean=data_info['stat'][0], std=data_info['stat'][1])]

    if args.cm:
        print(Fore.YELLOW + '*' * 80)
        logger.debug('Cutmix used')
        print('*' * 80 + Style.RESET_ALL)
    if args.mu:
        print(Fore.YELLOW + '*' * 80)
        logger.debug('Mixup used')
        print('*' * 80 + Style.RESET_ALL)
    if args.ra > 1:
        print(Fore.YELLOW + '*' * 80)
        logger.debug(f'Repeated Aug({args.ra}) used')
        print('*' * 80 + Style.RESET_ALL)

    '''
        Data Augmentation
    '''
    augmentations = []

    augmentations += [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(data_info['img_size'], padding=4)
    ]



    augmentations += [
        transforms.ToTensor(),
        *normalize]

    if args.re > 0:
        from utils.random_erasing import RandomErasing
        print(Fore.YELLOW + '*' * 80)
        logger.debug(f'Random erasing({args.re}) used ')
        print('*' * 80 + Style.RESET_ALL)

        augmentations += [
            RandomErasing(probability=args.re, sh=args.re_sh, r1=args.re_r1, mean=data_info['stat'][0])
        ]

    augmentations = transforms.Compose(augmentations)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, num_workers=args.workers, pin_memory=True,
        batch_sampler=RASampler(len(train_dataset), args.batch_size, 1, args.ra, shuffle=True, drop_last=True))
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    '''
        Training
    '''

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(args, optimizer, len(train_loader))

    #summary(model, (3, data_info['img_size'], data_info['img_size']))

    print()
    print("Beginning training")
    print()

    lr = optimizer.param_groups[0]["lr"]
    best_loss = -float('inf')
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        final_epoch = args.epochs
        args.epochs = final_epoch - (checkpoint['epoch'] + 1)

    for epoch in tqdm(range(args.epochs)):
        lr = train(train_loader, model, criterion, optimizer, epoch, scheduler, args)
        avgloss = validate(val_loader, model, criterion, lr, args, epoch=epoch)
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        },
            os.path.join(save_path, 'checkpoint.pth'))

        logger_dict.print()

        if avgloss < best_loss:
            print('* Best model upate *')
            best_loss = avgloss

            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, os.path.join(save_path, 'best.pth'))

        print('*' * 80)
        print(Style.RESET_ALL)

        writer.add_scalar("Learning Rate", lr, epoch)

    print(Fore.RED + '*' * 80)
    print('*' * 80 + Style.RESET_ALL)
    torch.save(model.state_dict(), os.path.join(save_path, 'checkpoint.pth'))


def train(train_loader, model, criterion, optimizer, epoch, scheduler, args):
    model.train()
    loss_val, acc1_val = 0, 0
    n = 0

    for i, (images, target) in enumerate(train_loader):
        if (not args.no_cuda) and torch.cuda.is_available():
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        if args.model == 'vitautoencoder':
            loss, output  = model(images)
        elif args.model == 'vitmaskedautoencoder':
            loss, output, mask = model(images)
        elif args.model == 'vitmaskedvideoautoencoder':
            loss, output  = model(images)

        elif args.model == 'vitmaskedvideoencoderwithhead':
             output  = model(images)
             target = target.to(torch.int64)
             target = target[:, 0] #for all N patches we just need one label for vidoes
             loss = criterion(output, target)

        n += images.size(0)
        loss_val += float(loss.item() * images.size(0))


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        if args.print_freq >= 0 and i % args.print_freq == 0:
            avg_loss= (loss_val / n)
            progress_bar(i, len(train_loader),
                         f'[Epoch {epoch + 1}/{args.epochs}][T][{i}]   Loss: {avg_loss:.4e}   LR: {lr:.7f}' + ' ' * 10)

    logger_dict.update(keys[0], avg_loss)
    writer.add_scalar("Loss/train", avg_loss, epoch)

    return lr


def validate(val_loader, model, criterion, lr, args, epoch=None):
    model.eval()
    loss_val, acc1_val = 0, 0
    n = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            if (not args.no_cuda) and torch.cuda.is_available():
                images = images.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

            if args.model == 'vitautoencoder':
                loss, output = model(images)
            elif args.model == 'vitmaskedautoencoder':
                loss, output, mask = model(images)
            #loss = criterion(output, target)
            elif args.model == 'vitmaskedvideoautoencoder':
                loss, output = model(images)
            elif args.model == 'vitmaskedvideoencoderwithhead':
                output = model(images)
                target = target.to(torch.int64)
                target = target[:, 0]  # for all N patches we just need one label for vidoes
                loss = criterion(output, target)
            #acc = accuracy(output, target, (1, 5))
            #acc1 = acc[0]
            n += images.size(0)
            loss_val += float(loss.item() * images.size(0))
            #acc1_val += float(acc1[0] * images.size(0))

            if args.print_freq >= 0 and i % args.print_freq == 0:
                avg_loss = (loss_val / n)
                progress_bar(i, len(val_loader),
                             f'[Epoch {epoch + 1}][V][{i}]   Loss: {avg_loss:.4e}    LR: {lr:.6f}')
    print()

    print(Fore.BLUE)
    print('*' * 80)

    logger_dict.update(keys[2], avg_loss)


    writer.add_scalar("Loss/val", avg_loss, epoch)


    return avg_loss


if __name__ == '__main__':
    parser = init_parser()
    args = parser.parse_args()
    global save_path
    global writer

    # random seed

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    np.random.seed(args.seed)  # Numpy module.
    random.seed(args.seed)  # Python random module.
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    model_name = args.model

    if not args.is_SPT:
        model_name += "-Base"
    else:
        model_name += "-SPT"

    if args.is_LSA:
        model_name += "-LSA"

    model_name += f"-{args.tag}-{args.dataset}-LR[{args.lr}]-Seed{args.seed}"
    save_path = os.path.join(os.getcwd(), 'save', model_name)
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    writer = SummaryWriter(os.path.join(os.getcwd(), 'tensorboard', model_name))

    # logger

    log_dir = os.path.join(save_path, 'history.csv')
    logger = log.getLogger(__name__)
    formatter = log.Formatter('%(message)s')
    streamHandler = log.StreamHandler()
    fileHandler = log.FileHandler(log_dir, 'a')
    streamHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)
    logger.setLevel(level=log.DEBUG)

    global logger_dict
    global keys

    logger_dict = Logger_dict(logger, save_path)
    keys = ['T Loss', 'T Top-1', 'V Loss', 'V Top-1']

    main(args)

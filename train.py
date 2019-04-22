# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import matplotlib
matplotlib.use('agg')
import time
import os
from model import ft_net
from random_erasing import RandomErasing
import yaml
from shutil import copyfile
import errno
from torch.utils.data import DataLoader

# Options
def option():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--gpu_ids', default='0,1', type=str)
    parser.add_argument('--data_dir',
                        default='/home/xiaoxi.xjl/re_id_dataset/Market-1501-v15.09.15/pytorch',
                        type=str, help='training dir path')
    parser.add_argument('--checkpoint_dir',
                        default='/home/xiaoxi.xjl/others_code/large-scale/Person_reID_baseline_pytorch',
                        type=str, help='training dir path')
    parser.add_argument('--checkpoint_fold',
                        default='debug-v201904221343',
                        type=str, help='training dir path')

    parser.add_argument('--color_jitter', default=True, help='use color jitter in training')
    parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
    parser.add_argument('--max_epoch', default=60, type=int, help='max epoch')
    parser.add_argument('--save_frequency', default=10, type=int)

    parser.add_argument('--stride', default=2, type=int, help='stride')
    parser.add_argument('--erasing_p', default=0.5, type=float,
                        help='Random Erasing probability, in [0,1]')
    parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
    parser.add_argument('--droprate', default=0.5, type=float, help='drop rate')
    opt = parser.parse_args()
    return opt

def main():
    # args
    opt = option()

    # GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
    cudnn.benchmark = True

    # transform
    transform_train_list = [
        transforms.Resize((256, 128), interpolation=3),
        transforms.Pad(10),
        transforms.RandomCrop((256, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    # random erasing
    if opt.erasing_p > 0:
        transform_train_list = transform_train_list + \
                               [RandomErasing(probability=opt.erasing_p,
                                              mean=[0.0, 0.0, 0.0])]

    # color jittering
    if opt.color_jitter:
        transform_train_list = [transforms.ColorJitter
                                (brightness=0.1, contrast=0.1, saturation=0.1,
                                                       hue=0)] \
                               + transform_train_list

    print(transform_train_list)

    data_transforms = {
        'train': transforms.Compose(transform_train_list),
    }

    # prepare dataset
    image_datasets = {}
    image_datasets['train'] = datasets.ImageFolder(os.path.join(opt.data_dir, 'train_all' ),
                                                   data_transforms['train'])

    # dataloader
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                                  shuffle=True, num_workers=8, pin_memory=True)
                   # 8 workers may work faster
                   for x in ['train']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train']}
    class_names = image_datasets['train'].classes

    use_gpu = torch.cuda.is_available()

    # ResNet50
    model = ft_net(len(class_names), opt.droprate, opt.stride)

    ignored_params = list(map(id, model.classifier.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    optimizer_ft = optim.SGD([
        {'params': base_params, 'lr': 0.1 * opt.lr},
        {'params': model.classifier.parameters(), 'lr': opt.lr}
    ], weight_decay=5e-4, momentum=0.9, nesterov=True)

    # Decay LR by a factor of 0.1 every 40 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=40, gamma=0.1)

    # Train
    dir_name = os.path.join(opt.checkpoint_dir, opt.checkpoint_fold)
    if dir_name is not None:
        mkdir_if_missing(dir_name)

    # record every run
    copyfile('./train.py', dir_name + '/train.py')
    copyfile('./model.py', dir_name + '/model.py')

    # save opts
    with open('%s/opts.yaml' % dir_name, 'w') as fp:
        yaml.dump(vars(opt), fp, default_flow_style=False)

    # model to gpu
    model = nn.DataParallel(model).cuda()

    criterion = nn.CrossEntropyLoss()

    train_model(dataloaders,opt,use_gpu,dataset_sizes,model, criterion, optimizer_ft, exp_lr_scheduler,
                        num_epochs=opt.max_epoch, save_dir=dir_name)

    return

# train
def train_model(dataloaders,opt,use_gpu,dataset_sizes,model,
                criterion, optimizer, scheduler,save_dir, num_epochs=25):
    y_loss = {}
    y_loss['train'] = []
    y_err = {}
    y_err['train'] = []
    begin_time=time.time()
    for epoch in range(num_epochs):
        since = time.time()
        print('Epoch {}/{}'.format(epoch+1, num_epochs ))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode

            running_loss = 0.0
            running_corrects = 0.0
            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data
                now_batch_size,c,h,w = inputs.shape
                if now_batch_size<opt.batchsize: # skip the last batch
                    continue

                if use_gpu:
                    inputs = Variable(inputs.cuda().detach())
                    labels = Variable(labels.cuda().detach())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()

                # forward
                outputs = model(inputs)

                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)


                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item() * now_batch_size
                running_corrects += float(torch.sum(preds == labels.data))

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1.0-epoch_acc)

            # save the model
            if (epoch+1) % opt.save_frequency == 0:
                save_network(model, epoch+1,save_dir)

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print()

    time_elapsed = time.time() - begin_time
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

# save model
def save_network(network, epoch_label,save_dir):
    save_filename = 'net_%s.pth'% epoch_label
    save_path = os.path.join(save_dir,save_filename)
    torch.save(network.module.state_dict(), save_path)

# mkdir if missing
def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
            print("Make directory:{}".format(directory))
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

if __name__ == '__main__':
    main()











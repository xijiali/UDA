# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from torchvision import datasets, transforms

import os
import scipy.io

from model import ft_net_dense,ft_net

# Options
def option():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--gpu_ids', default='0,1', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
    parser.add_argument('--test_dir',
                        default='/home/xiaoxi.xjl/re_id_dataset/Market-1501-v15.09.15/pytorch',
                        type=str,)
    parser.add_argument('--checkpoint_dir',
                        default='/home/xiaoxi.xjl/others_code/large-scale/Person_reID_baseline_pytorch/debug-v201904221343',
                        type=str)
    parser.add_argument('--file_name',
                        default='net_59.pth',
                        type=str)
    parser.add_argument('--save_dir',
                        default='/home/xiaoxi.xjl/others_code/large-scale/Person_reID_baseline_pytorch/debug-v201904221343',
                        type=str)
    parser.add_argument('--save_name',
                        default='net_59_feature.mat',
                        type=str)
    parser.add_argument('--batchsize', default=256, type=int, help='batchsize')
    parser.add_argument('--stride', default=2, type=int)
    parser.add_argument('--nclasses', default=751, type=int, help='batchsize')
    parser.add_argument('--multi', action='store_true', default=False, help='use multiple query')

    opt = parser.parse_args()

    return opt

def main():
    opt=option()
    data_dir = opt.test_dir
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
    cudnn.benchmark = True

    # transform
    data_transforms = transforms.Compose([
        transforms.Resize((256, 128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # prepare dataset
    image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms)
                      for x in ['gallery','query']}

    # dataloader
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                  batch_size=opt.batchsize,
                                                  shuffle=False, num_workers=16)
                   for x in ['gallery', 'query']}
    class_names = image_datasets['query'].classes
    use_gpu = torch.cuda.is_available()

    gallery_path = image_datasets['gallery'].imgs
    query_path = image_datasets['query'].imgs
    gallery_cam, gallery_label = get_id(gallery_path)
    query_cam, query_label = get_id(query_path)

    # Load Collected data Trained model
    print('-------test-----------')
    model_structure = ft_net(opt.nclasses)

    checkpoint_dir=os.path.join(opt.checkpoint_dir,opt.file_name)
    model = load_network(model_structure,checkpoint_dir)

    # Remove the final fc layer and classifier layer
    model.classifier.classifier = nn.Sequential()

    # Change to test mode
    model = model.eval()
    if use_gpu:
        model = model.cuda()

    # Extract feature
    with torch.no_grad():
        gallery_feature = extract_feature(model, dataloaders['gallery'])
        query_feature = extract_feature(model, dataloaders['query'])

    # Save to Matlab for check
    result = {'gallery_f': gallery_feature.numpy(),
              'gallery_label': gallery_label,
              'gallery_cam': gallery_cam,
              'query_f': query_feature.numpy(),
              'query_label': query_label,
              'query_cam': query_cam}
    save_file=os.path.join(opt.save_dir,opt.save_name)
    scipy.io.savemat(save_file, result)

def load_network(network,save_path):
    network.load_state_dict(torch.load(save_path))
    return network

# Extract feature

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_feature(model,dataloaders):
    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        count += n
        ff = torch.FloatTensor(n,512).zero_()

        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            outputs = model(input_img) 
            f = outputs.data.cpu().float()
            ff = ff+f
        # norm feature
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features,ff), 0)
    return features

def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        #filename = path.split('/')[-1]
        filename = os.path.basename(path)
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels

if __name__ == '__main__':
    main()




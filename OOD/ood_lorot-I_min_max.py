#!/usr/bin/env python
from __future__ import print_function
import argparse
import os
import sys
from pprint import pprint
import torch.backends.cudnn as cudnn
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import datasets.datasets as datasets
from datasets.ood_datasets import get_dataset, get_superclass_list, get_subclass_dataset
from torch.utils.data import DataLoader
from resnet import ResNet34, ResNet18
import copy
from torch.utils.data.dataset import Subset
from torchvision import transforms# datasets,

parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs to train for [default: 10]')
parser.add_argument('--lr', '--learning_rate', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--batch_size', default=64, type=int, help='train batchsize')
parser.add_argument('--iter', default='False', type=str)
parser.add_argument('--dataset', type=str, default='cifar10', help="mnist | svhn | cifar10 | cifar100 | tiny_imagenet")
parser.add_argument('--out-dataset', type=str, default='svhn', help="mnist | svhn | cifar10 | cifar100 | tiny_imagenet")
parser.add_argument('--dataroot', type=str, default='./data')
parser.add_argument('--outf', type=str, default='./log')
parser.add_argument('--loss', type=str, default='ce')
parser.add_argument('--r_ratio', default=0.5, type=float, help='self-supervised loss ratio')
parser.add_argument('--workers', default=6, type=int,
                    help="number of data loading workers (default: 4)")
parser.add_argument('--model', type=str, default='classifier32', help='classifier32 | WRN')
parser.add_argument("--ood_dataset", help='Datasets for OOD detection',
                    default=None, nargs="*", type=str)
parser.add_argument('--trial', default=0, type=int)
parser.add_argument('--psize', default=2, type=int,
                        help='patch size_min')
parser.add_argument('--maxpsize', default=16, type=int,
                    help='patch size_min')
args = parser.parse_args()
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
def kl_div(d1, d2):
    """
    Compute KL-Divergence between d1 and d2.
    """
    dirty_logs = d1 * torch.log2(d1 / d2)
    return torch.sum(torch.where(d1 != 0, dirty_logs, torch.zeros_like(d1)), axis=1)

def rand_bbox(size, psize, maxpsize):
    W = size[2]
    H = size[3]

    r = np.random.randint(psize, maxpsize + 1)

    # uniform
    cx = np.random.randint(W-r)
    cy = np.random.randint(H-r)

    bbx1 = cx
    bby1 = cy
    bbx2 = cx + r
    bby2 = cy + r

    return bbx1, bby1, bbx2, bby2

def train(epoch,net,optimizer,train_loader, iter):
    net.train()
    for batch_idx, (inputs_x, labels_x) in enumerate(train_loader):
        batch_size = inputs_x.size(0)
        inputs, labels_x = inputs_x.cuda(), labels_x.cuda()
        bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), args.psize, args.maxpsize)
        idx2 = torch.randint(4, size=(inputs.size(0),))
        for i in range(inputs.size(0)):
            inputs[i][:, bbx1:bbx2, bby1:bby2] = torch.rot90(inputs[i][:, bbx1:bbx2, bby1:bby2], idx2[i], [1, 2])
        rotlabel = idx2
        rotlabel = rotlabel.cuda()
        logits, rotlogits = net(inputs, both=True)
        loss_x = criterion(logits, labels_x)
        rotloss = criterion(rotlogits, rotlabel)
        loss = loss_x + rotloss * args.r_ratio

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sys.stdout.write('\r')

        sys.stdout.write('%s: | Epoch [%3d/%3d] loss_x: %.2f, gr_loss: %.2f'
                         % (args.dataset, epoch, args.num_epochs, loss_x.item(), rotloss.item()))
        sys.stdout.flush()

    return iter

def test(net1, d):
    net1.eval()
    prediction1 = torch.zeros(50000)
    labels = torch.zeros(50000)
    total = 0
    total2 = 0
    open_labels = torch.ones(50000)
    probs = torch.zeros(50000)

    n = 0
    with torch.no_grad():
        for batch_idx, (inputs, label) in enumerate(testloader):
            bsz = inputs.size(0)
            inputs, label = inputs.cuda(), label.long().cuda()

            o1 = net1(inputs)
            pred1 = torch.softmax(o1, dim=1).max(1)[1]
            classification_smax = F.softmax(o1, dim=1)
            class_uniform_dist = torch.ones_like(classification_smax) * 0.1
            _1 = -1 * kl_div(class_uniform_dist, classification_smax)

            total += bsz
            total2 += bsz
            for b in range(bsz):
                probs[n] = _1[b]
                labels[n] = label[b]
                prediction1[n] = pred1[b]
                n+=1

    # openset test
    open_labels[total:] = 0
    with torch.no_grad():
        for batch_idx, (inputs, label) in enumerate(ood_test_loader[d]):
            bsz = inputs.size(0)
            inputs, label = inputs.cuda(), label.long().cuda()

            o1 = net1(inputs)
            pred1 = torch.softmax(o1, dim=1).max(1)[1]
            classification_smax = F.softmax(o1, dim=1)
            class_uniform_dist = torch.ones_like(classification_smax) * 0.1
            _1 = -1 * kl_div(class_uniform_dist, classification_smax)

            total2 += bsz
            for b in range(bsz):
                probs[n] = _1[b]
                labels[n] = label[b]
                n += 1
    prob = probs[:total2].reshape(-1, 1)
    prob[prob == -float('inf')] = -10000
    auc = roc_auc_score(open_labels[:total2].cpu().numpy(), prob)
    correct1 = prediction1[:total].eq(labels[:total]).sum().item()
    acc1 = correct1 / total
    return acc1, auc

acc_results = []
auc_results = []
criterion = nn.CrossEntropyLoss()
options = vars(args)
options['dataroot'] = os.path.join(options['dataroot'], options['dataset'])
if not os.path.exists('./logs'):
    os.makedirs('./logs')
    if not os.path.exists('./logs/ood'):
        os.makedirs('./logs/ood')
if not os.path.exists('./logs/ood/'+ args.dataset):
    os.makedirs('./logs/ood/'+ args.dataset)

stats_log = open('./logs/ood/' + args.dataset + '/%d_psize_lorotI_%.2f_%d_%d' % (args.trial, args.r_ratio, args.psize, args.maxpsize) + '.txt', 'w')


use_gpu = torch.cuda.is_available()
options.update(
    {

        'use_gpu': use_gpu
    }
)
out_dataset = datasets.create(options['out_dataset'], **options)
dataset = datasets.create(options['dataset'], **options)
trainloader, testloader = dataset.trainloader, dataset.testloader
outloader = out_dataset.testloader
options.update(
    {
        'num_classes': dataset.num_classes
    }
)

if options['ood_dataset'] is None:
    if options['dataset'] == 'cifar10':
        options['ood_dataset'] = ['svhn', 'lsun_resize', 'imagenet_resize', 'lsun_fix', 'imagenet_fix', 'cifar100']#, 'interp']
        options['image_size'] = (32, 32, 3)
    elif options['dataset'] == 'imagenet':
        options['ood_dataset'] = ['cub', 'stanford_dogs', 'flowers102', 'places365', 'food_101', 'caltech_256', 'dtd', 'pets']
        options['num_classes'] = 30
ood_eval = True
ood_test_loader = dict()
kwargs = {'pin_memory': False, 'num_workers': 40}
for ood in options['ood_dataset']:
    ood_test_set = get_dataset(options, dataset=ood, test_only=True, image_size=options['image_size'], eval=ood_eval)
    ood_test_loader[ood] = DataLoader(ood_test_set, shuffle=False, batch_size=options['batch_size'], **kwargs)


model = ResNet18(options['num_classes'], 4).cuda()
model = torch.nn.DataParallel(model)
cudnn.benchmark=True

optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))

bestauc = 0
iter = 0
aucs = []
accs = []
for epoch in range(args.num_epochs + 1):
    lr = args.lr
    if epoch >= 50:
        lr /= 10

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    iter = train(epoch, model, optimizer, trainloader, iter)
    if epoch % 10 == 0:
        aucs = []
        accs = []
        if epoch % 100 == 0:
            aucs = []
            accs = []
            for d in options['ood_dataset']:
                acc, auc = test(model, d)
                aucs.append(auc)
                accs.append(acc)
                print("E[%d] [%s] AUC Err: [%.3f] ACC : [%.2f]\n" % (epoch, d, auc * 100, acc * 100))
                # if epoch % 100 == 0:
                stats_log.write("--Epoch %d--\n" % epoch)
                stats_log.write("E[%d] [%s] AUC Err : [%.3f] ACC : [%.2f]\n" % (epoch, d, auc* 100, acc * 100))
        # auc_results.append(aucs)
        # acc_results.append(accs)
# data_list = options['ood_dataset']
# for j in range(len(data_list)):#dataset
#     ood_Dataset = data_list[j]
#     print('OOD Dataset : {}'.format(ood_Dataset))
#     stats_log.write('\n\nOOD Dataset : {}\n'.format(ood_Dataset))
#     # stats_log.flush()
#     for i in range(args.num_epochs // 10 + 1):
#         print('Epoch [%d] acc1 : [%.3f] ~ auc1 Err: [%.3f]' % (i * 10, acc_results[i][j]* 10, auc_results[i][j]))
#         stats_log.write('Epoch [%d] acc1 : [%.3f] ~ auc1 Err : [%.3f]' % (i * 10, acc_results[i][j]* 10, auc_results[i][j]))
#         # stats_log.write(
#         #     'Epoch [%d] acc1 : [%.3f] acc2 : [%.3f] ~ auc1 : [%.3f] auc2 : [%.3f] auc3 : [%.3f] auc4 : [%.3f]' % (
#         #     i * 10, acc_results[i][2 * j] * 10, acc_results[i][2 * j + 1]* 10, auc_results[i][4 * j], auc_results[i][4 * j + 1],
#         #     auc_results[i][4 * j + 2], auc_results[i][4 * j + 3]))
#         stats_log.flush()
# stats_log.close()

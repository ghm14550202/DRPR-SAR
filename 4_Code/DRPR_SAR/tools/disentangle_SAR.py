from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from copy import deepcopy
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import wandb
import os
import time
import argparse
import datetime
from torch.autograd import Variable
import pdb
import sys

sys.path.append('.')
from utils.SAR_loader import get_dataloader
from utils.set import *
from networks_V1.rot_vqvae_SAR import RotVQVAE_SAR_Res






def reconst_images(epoch=2, batch_size=64, batch_num=2, dataloader=None, model=None):
    SAR_dataloader = dataloader

    model.eval()

    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(SAR_dataloader):
            if batch_idx >= batch_num:
                break
            else:
                X, y = X.cuda(), y.cuda().view(-1, )
                _, _, _, gx, _, _ = model(X)

                grid_X = torchvision.utils.make_grid(X[:batch_size].data, nrow=4, padding=2, normalize=False)
                wandb.log({"_Batch_{batch}_X.jpg".format(batch=batch_idx): [
                    wandb.Image(grid_X)]}, commit=False)
                grid_GX = torchvision.utils.make_grid(gx[:batch_size].data, nrow=4, padding=2, normalize=False)
                wandb.log({"_Batch_{batch}_GX.jpg".format(batch=batch_idx): [
                    wandb.Image(grid_GX)]}, commit=False)
                grid_RX = torchvision.utils.make_grid((X[:batch_size] - gx[:batch_size]).data, nrow=4, padding=2,
                                                        normalize=False)
                wandb.log({"_Batch_{batch}_RX.jpg".format(batch=batch_idx): [
                    wandb.Image(grid_RX)]}, commit=False)
    print('reconstruction complete!')


def test(epoch, model, testloader):
    # set model as testing mode
    model.eval()
    # The VQ model has no Gaussian posterior statistics.
    acc_avg = AverageMeter()
    sparse_avg = AverageMeter()
    top1 = AverageMeter()
    commit_avg = AverageMeter()

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(testloader):
            # distribute data to device
            x, y = x.cuda(), y.cuda().view(-1, )
            bs = x.size(0)

            x_flatten = x.view(bs, -1)  # Flatten each image to a vector
            # norm = torch.norm(x_flatten, p=2, dim=1)  # Compute the L2 norm for each image
            # norm = norm.unsqueeze(1).unsqueeze(2).unsqueeze(3)  # [bs, 1, 1, 1]

            # norm = torch.norm(torch.abs(x.view(bs -1)), p=2, dim=1)
            _, out, z_q, gx, commit_loss, _ = model(x)

            # 移除归一化操作，因为数据已在dataloader中归一化
            acc_gx = 1 - F.mse_loss(gx, x, reduction='sum') / 100
            acc_rx = 1 - F.mse_loss(x - gx, x, reduction='sum') / 100
            # 原来的代码使用了归一化:
            # acc_gx = 1 - F.mse_loss(torch.div(gx, norm),
            #                         torch.div(x, norm),
            #                         reduction='sum') / 100
            # acc_rx = 1 - F.mse_loss(torch.div(x - gx, norm),
            #                         torch.div(x, norm),
            #                         reduction='sum') / 100

            acc_avg.update(acc_gx.data.item(), bs)
            # measure accuracy and record loss
            sparse_avg.update(acc_rx.data.item(), bs)
            # measure accuracy and record loss
            prec1, _, _, _ = accuracy(out.data, y.data, topk=(1, 5))
            top1.update(prec1.item(), bs)

            commit_avg.update(commit_loss.mean().item(), bs)

            wandb.log({'acc_avg': acc_avg.avg, \
                       'sparse_avg': sparse_avg.avg, \
                       'test-RX-acc': top1.avg, \
                   'test-commit': commit_avg.avg}, commit=False)
        # plot progress
        print("\n| Validation Epoch #%d\t\tRec Acc: %.4f Class Acc: %.4f Commit: %.4f" %
              (epoch, acc_avg.avg, top1.avg, commit_avg.avg))
        reconst_images(epoch=epoch, batch_size=64, batch_num=2, dataloader=testloader, model=model)
        torch.save(model.state_dict(),
                   os.path.join(args.save_dir, 'model_epoch{}.pth'.format(epoch + 1)))  # save motion_encoder
        print("Epoch {} model saved!".format(epoch + 1))


def train(args, epoch, model, optimizer, trainloader):
    model.train()
    model.training = True

    loss_avg = AverageMeter() #平均损失
    loss_rec = AverageMeter()  # 重建损失
    loss_ce = AverageMeter()  #交叉损失
    loss_commit = AverageMeter()
    top1 = AverageMeter()

    print('\n=> Training Epoch #%d, LR=%.4f' % (epoch, optimizer.param_groups[0]['lr']))
    for batch_idx, (x, y) in enumerate(trainloader):
        # x, y, y_b, lam, mixup_index = mixup_data(x, y, alpha=args.alpha)
        # x, y, y_b = x.cuda(), y.cuda().view(-1, ), y_b.cuda().view(-1, )
        # x, y = Variable(x), [Variable(y), Variable(y_b)]

        x, y = x.cuda(), y.cuda().view(-1, )
        x, y = Variable(x), Variable(y)
        bs = x.size(0)
        optimizer.zero_grad()

        _, out, z_q, xi, commit_loss, _ = model(x)

        if args.curriculum:
            if epoch < 100:
                re = 10*args.re
            elif epoch < 200:
                re = 8*args.re
            else:
                re = 7*args.re
        else:
            re = args.re

        # if args.curriculum:
        #     re = args.re

        l1 = F.mse_loss(xi, x)
        # cross_entropy = lam * F.cross_entropy(out, y[0]) + (1. - lam) * F.cross_entropy(out, y[1])
        cross_entropy = F.cross_entropy(out, y)
        l2 = cross_entropy
        # The VQ-VAE has no Gaussian posterior. Use its commitment loss
        # in the same weighted loss position as the original KL term.
        l3 = commit_loss.mean()
        loss = re * l1 + args.ce * l2 + args.kl * l3
        loss.backward()
        optimizer.step()


        # prec1, prec5, correct, pred = accuracy(out.data, y[0].data, topk=(1, 5))
        prec1, prec5, correct, pred = accuracy(out.data, y.data, topk=(1, 5))
        loss_avg.update(loss.data.item(), bs)
        loss_rec.update(l1.data.item(), bs)
        loss_ce.update(cross_entropy.data.item(), bs)
        loss_commit.update(l3.data.item(), bs)
        top1.update(prec1.item(), bs)

        n_iter = (epoch - 1) * len(trainloader) + batch_idx
        wandb.log({'loss': loss_avg.avg, \
                   'loss_rec': loss_rec.avg, \
                   'loss_ce': loss_ce.avg, \
                   'loss_commit': loss_commit.avg, \
                   'acc': top1.avg,
                   're_weight': re,
                   'lr':optimizer.param_groups[0]['lr']}, step=n_iter)
        if (batch_idx + 1) % 30 == 0:
            sys.stdout.write('\r')
            sys.stdout.write(
                '| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Loss_rec: %.4f Loss_ce: %.4f Loss_commit: %.4f Acc@1: %.3f%%'
                % (epoch, args.epochs, batch_idx + 1,
                   len(trainloader), loss_avg.avg, loss_rec.avg, loss_ce.avg, loss_commit.avg, top1.avg))


def main(args):
    learning_rate = 1.e-4  #1.e-3
    learning_rate_min = 2.e-5  #2.e-4
    CNN_embed_dim = args.dim
    feature_dim = args.fdim
    setup_logger(args.save_dir)
    use_cuda = torch.cuda.is_available()
    best_acc = 0
    print('\n[Phase 1] : Data Preparation')

    train_path = './data/%s/train' % (args.dataset)
    test_path = './data/%s/test' % (args.dataset)

    if args.dataset == 'mstar':
        num_class = 10
        mean = (0.13620707)
        std = (0.11149936)
        print("| Preparing mstar dataset...")
        sys.stdout.write("| ")
    elif args.dataset == 'acd':
        num_class = 6
        mean = (0.0565084)
        std = (0.09828296)
        print("| Preparing acd dataset...")
        sys.stdout.write("| ")
    elif args.dataset == 'opensar':
        num_class = 6
        mean = (0.05527896)
        std = (0.06979666)
        print("| Preparing opensar dataset...")
        sys.stdout.write("| ")
    elif args.dataset == 'FUSAR':
        num_class = 10
        mean = (0.10994489,)
        std = (0.10608266,)
        print("| Preparing FUSAR dataset...")
        sys.stdout.write("| ")
    elif args.dataset == 'mstarcolor':
        num_class = 10
        mean= (0.26255840, 0.29695627, 0.86251346)
        std = (0.07295840, 0.13998037, 0.11734496)
        print("| Preparing mstarcolor dataset...")
        sys.stdout.write("| ")

    train_loader, test_loader = get_dataloader(args.dataset, args.batch_size, args.size, train_path=train_path,
                                               test_path=test_path,  mean=mean, std=std)
    # Model: keep the original stage-1 residual-classification objective,
    # replacing only the Gaussian VAE bottleneck with rotation-trick VQ.
    print('\n[Phase 2] : Model setup')
    if args.backbone != 'resnet':
        raise ValueError(
            "The rotation-trick implementation currently provides only "
            "RotVQVAE_SAR_Res; use --backbone resnet."
        )
    model = RotVQVAE_SAR_Res(
        d=feature_dim,
        z=CNN_embed_dim,
        num_classes=num_class,
        with_classifier=True,
        classifier_mode='residual',
        num_embeddings=args.num_embeddings,
        commitment_weight=args.commitment_weight,
        decay=args.ema_decay,
        threshold_ema_dead_code=args.threshold_ema_dead_code,
    )

    if use_cuda:
        model.cuda()
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    optimizer = AdamW([
        {'params': model.parameters()}
    ], lr=learning_rate, betas=(0.9, 0.999), weight_decay=1.e-6)

    if args.optim == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50,
                                                        eta_min=learning_rate_min)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.step, gamma=0.1, last_epoch=-1)

    print('\n[Phase 3] : Training model')
    print('| Training Epochs = ' + str(args.epochs))

    start_epoch = 1
    elapsed_time = 0
    for epoch in range(start_epoch, start_epoch + args.epochs):
        start_time = time.time()
        train(args, epoch, model, optimizer, train_loader)
        scheduler.step()
        if epoch % 10 == 0:
            test(epoch, model, test_loader)

        epoch_time = time.time() - start_time
        elapsed_time += epoch_time

    wandb.finish()
    print('\n[Phase 4] : Testing model')
    print('* Test results : Acc@1 = %.2f%%' % (best_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch SAR Training')
    parser.add_argument('--save_dir', default='./results_SAR/autoaug_new_8_0.5/', type=str, help='save_dir')
    parser.add_argument('--seed', default=666, type=int, help='seed')
    parser.add_argument('--dataset', default='mstar', type=str, help='dataset = [mstar/acd/opensar/FUSAR]')
    parser.add_argument('--optim', default='cosine', type=str, help='optimizer')
    parser.add_argument('--alpha', default=2.0, type=float, help='mix up')

    parser.add_argument('--epochs', default=300, type=int, help='training_epochs')
    parser.add_argument('--size', default=128, type=int)
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--dim', default=2048, type=int, help='CNN_embed_dim')#2048
    parser.add_argument('--T', default=50, type=int, help='Cosine T')
    parser.add_argument('--fdim', default=32, type=int, help='featdim')#32
    parser.add_argument('--step', nargs='+', type=int)
    parser.add_argument('--re', default=1.0, type=float, help='reconstruction weight')
    parser.add_argument('--curriculum', default=True,
                        help='Curriculum for reconstruction term which helps for better convergence')
    parser.add_argument('--kl', default=1, type=float,
                        help='weight of the VQ commitment loss (legacy KL argument)')
    parser.add_argument('--ce', default=0.2, type=float, help='cross entropy weight')  #较大的 𝛾 γ 会增加分类任务的权重。较小的 𝛾 则更注重生成任务的优化。
    parser.add_argument('--backbone', default='resnet', type=str, choices=['resnet'],
                        help='rotation-trick VQ-VAE backbone')
    parser.add_argument('--num_embeddings', default=512, type=int,
                        help='number of vectors in the VQ codebook')
    parser.add_argument('--commitment_weight', default=1.0, type=float,
                        help='encoder commitment-loss weight inside the VQ layer')
    parser.add_argument('--ema_decay', default=0.8, type=float,
                        help='EMA decay for the VQ codebook')
    parser.add_argument('--threshold_ema_dead_code', default=0, type=float,
                        help='reactivate codebook entries below this EMA usage threshold')
    args = parser.parse_args()
    wandb.init(config=args, name=args.save_dir.replace("results/", ''), mode="offline")
    set_random_seed(args.seed)
    main(args)

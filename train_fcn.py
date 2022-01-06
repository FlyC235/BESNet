# Author: Fenglei Chen
# Email: chenfl1201@foxmail.com

import os
import sys
import math
import time
import shutil
import datetime
import argparse
from types import new_class
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from libs.models import *
from libs.losses import loss
from libs.utils.metrics import Evaluator
from libs.utils.lr_scheduler import LR_Scheduler_Head
from libs.datasets import TrainLoader, TestLoader

from albumentations import Compose, Rotate, Flip

categories = [
    'Impervious surface',
    'Building',
    'Low vegetation',
    'Tree',
    'Car',
    'Clutter/Background'
]

class SegmentTrainer(object):
    def __init__(self, args, definetime=None):

        # set datetime
        if not definetime:
            dtime = datetime.datetime.now().strftime(f"%Y-%m-%d_%H-%M-%S")
        else:
            dtime = definetime

        exp_name = '_'.join([args.model, str(args.learning_rate), str(args.batch_size)])
        self.exp_dir = os.path.join(args.save_dir, args.dataset, exp_name, dtime)

        if not os.path.exists(self.exp_dir):
            os.makedirs(self.exp_dir)

        # save model checkpoint
        self.checkpoint_dir = os.path.join(self.exp_dir, 'checkpoint')
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)

        # runtime gpu environment
        print(f"Available device: {torch.cuda.device_count()}")
        print(f"Current cuda device: {torch.cuda.current_device()}")

        model = eval(args.model)(nclass=args.num_classes, backbone=args.backbone , aux=args.aux_classifier)
        
        # set optimizer
        self.optimizer = optim.SGD(
            [{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': args.learning_rate}],
            lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

        self.model = model.cuda()

        self.criterion = loss.CriterionDSN(
            class_weight=None, 
            aux_classifier=args.aux_classifier, 
            loss_weight=args.auxloss_weight, 
            ignore_index=255,  
            reduction='mean'
            )
            
        self.testcriterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
        self.boundary_loss = loss.DetailAggregateLoss()

        # set data loader
        aug = Compose([
            Rotate(limit=(-30, 30)),
            Flip(always_apply=False, p=0.5),
            ])

        train_T = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4752, 0.3216, 0.3188], [0.2108, 0.1484, 0.1431]),
            ])

        train_set = TrainLoader.TrainDataset(
            root=args.data_dir, 
            data_list=args.train_list,
            transform=train_T, 
            aug_transform=aug)
        
        self.train_loader = data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=True,
            pin_memory=True)

        if not args.no_val:

            test_T = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4752, 0.3216, 0.3188], [0.2108, 0.1484, 0.1431]),
            ])

            test_set = TestLoader.TestDataset(
                root=args.data_dir, 
                data_list=args.test_list,
                transform=test_T,
                )

            self.test_loader = data.DataLoader(
                test_set,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True
                )

            print(f"test_set: {len(test_set)}")

        # Define LR_scheduler
        self.scheduler = LR_Scheduler_Head(
            mode=args.lr_scheduler, 
            base_lr=args.learning_rate,
            num_epochs=args.max_epoches, 
            iters_per_epoch=len(self.train_loader),
            warmup_epochs=args.warmup_epochs
            )

        # Define Evaluator
        self.evaluator = Evaluator(args.num_classes)
        
        # tensorboard
        self.writer = SummaryWriter(log_dir=self.exp_dir)

        self.best_metrics = {'Acc':0, 'mIoU':0, 'mF1':0, 'F1':None, 'epoch':0}
        self.best_loss = np.inf

        print(f"finish ok!")

    def train_phase(self, args, epoch):
        
        start = time.time()
        self.model.train()
        train_loss = list()
        loss_boundery_bce = list()
        loss_boundery_dice = list()
        tqdm_train_loader = tqdm(self.train_loader, total=len(self.train_loader))
        for num_iter, (images, gts, _) in enumerate(tqdm_train_loader):

            tqdm_train_loader.set_description(f'Model: {args.model} Epoch: {epoch}/{args.max_epoches}')
            images = images.cuda()
            gts = gts.cuda()
            self.optimizer.zero_grad()
            self.scheduler(self.optimizer, num_iter, epoch, self.best_metrics['Acc'])
            lr = self.optimizer.param_groups[0]['lr']
            preds = self.model(images)[0:2]
            # calculate loss
            loss = self.criterion(preds, gts)
            boundary_detail = self.model(images)[2]

            boundery_bce_loss, boundery_dice_loss = self.boundary_loss(boundary_detail, gts)
            
            loss = loss + boundery_bce_loss + 0.8*boundery_dice_loss
            loss.backward()
            self.optimizer.step()
            train_loss.append(loss.item())
            loss_boundery_bce.append(boundery_bce_loss.item())
            loss_boundery_dice.append(boundery_dice_loss.item())
            tqdm_train_loader.set_postfix(Loss=f"{loss.item():.5f}")
            
        mean_train_loss = np.mean(train_loss)
        mean_loss_boundery_bce = np.mean(loss_boundery_bce)
        mean_loss_boundery_dice = np.mean(loss_boundery_dice)
        runtime = time.time() - start

        print(f"Epoch: {epoch}/{args.max_epoches} || lr: {lr:.5f} ||"
                         f" Loss: {mean_train_loss:.5f} Boundery_Loss:{mean_loss_boundery_bce:.5f},{mean_loss_boundery_dice:.5f} Time: {math.floor(runtime//3600):2d}h:"
                         f"{math.floor(runtime%3600//60):2d}m:{math.floor(runtime%60):2d}s")
        
        if mean_train_loss < self.best_loss and args.no_val:
            print('Save model...')
            self.best_loss = mean_train_loss
            torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, 'best_loss.pth'))

        
    def test_phase(self, args, epoch):

        self.model.eval()
        self.evaluator.reset()
        test_loss = list()
        start = time.time()
        with torch.no_grad():
            tqdm_test_loader = tqdm(self.test_loader, total=len(self.test_loader))
            for num_iter, (images, gts, _) in enumerate(tqdm_test_loader):
                tqdm_test_loader.set_description(f'Model: {args.model} Epoch: {epoch}/{args.max_epoches}')
                images = images.cuda()
                gts = gts.cuda()
                preds  = self.model(images)[0]
                # necessary to upsample when test
                loss = self.testcriterion(preds, gts)

                tqdm_test_loader.set_postfix(Loss=f"{loss:.5f}",)
                test_loss.append(loss.item())

                preds = preds.data.cpu().numpy()
                gts = gts.cpu().numpy()
                preds = np.argmax(preds, axis=1)

                # Add batch sample into evaluator
                self.evaluator.add_batch(gts, preds)

            runtime = time.time() - start
            mean_test_loss = np.mean(test_loss)
            self.writer.add_scalar('test/test_epoch_loss', mean_test_loss, epoch)

            # Fast test during the training
            Acc = self.evaluator.Pixel_Accuracy()
            Acc_class = self.evaluator.Pixel_Accuracy_Class()
            mIoU = self.evaluator.Mean_Intersection_over_Union()
            F1, mF1 = self.evaluator.Calculate_F1_Score()

            self.writer.add_scalar('test/mIoU', mIoU, epoch)
            self.writer.add_scalar('test/Acc', Acc, epoch)
            self.writer.add_scalar('test/Acc_class', Acc_class, epoch)
            self.writer.add_scalar('test/mF1', mF1, epoch)

            print(f"Epoch: {epoch}/{args.max_epoches} || Loss: {mean_test_loss:.5f} || Acc: {Acc:.5f} || mIoU: {mIoU:.5f} || mF1: {mF1:.5f}"
                      f" || Time: {math.floor(runtime//3600):2d}h:{math.floor(runtime%3600//60):2d}m:{math.floor(runtime%60):2d}s")

            print(f"\n\nEvaluation Metrics:\nAcc: {Acc:.5f} \nAcc_class: {Acc_class:.5f} \nmIoU: {mIoU:.5f} \nmF1: {mF1:.5f}"
                     f"\nCategory F1: {dict(zip(categories, F1))}\n")

            if self.best_loss > mean_test_loss:
                self.best_loss = mean_test_loss
                torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, 'best_model.pth'))

            if self.best_metrics['Acc'] < Acc:
                self.best_metrics['Acc'] = Acc
                self.best_metrics['mIoU'] = mIoU
                self.best_metrics['mF1'] = mF1
                self.best_metrics['F1'] = dict(zip(categories, F1))
                self.best_metrics['epoch'] = epoch
                torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, 'best_metrics.pth'))
        
        with open(testing_results_file, 'a') as f:
            f.write(f"Epo:{epoch}   Acc: {Acc:.5f}   Acc_class: {Acc_class:.5f}   mIoU: {mIoU:.5f}   mF1: {mF1:.5f}"
                     f"\nCategory F1: {dict(zip(categories, F1))}\n")

    def train(self, args):
        cudnn.benchmark = True
        cudnn.enabled = True
        for epoch in range(args.max_epoches):
            print(f'train {epoch}/{args.max_epoches}')
            print('='*10)
            try:
                self.train_phase(args, epoch)
            except:
                e = sys.exc_info()[0]
                shutil.rmtree(self.exp_dir)
                print(f" Error {e} occur !")
                raise
            if not args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
                self.test_phase(args, epoch)

if __name__ == '__main__':

    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    def get_arguments():
        """
        Parse all the arguments
        Returns: args
        A list of parsed arguments.
        """
        parser = argparse.ArgumentParser(description="Remote sensing Competition")
        parser.add_argument("--batch_size", type=int, default=2,
                            help="Number of images sent to the network in one step.")
        parser.add_argument("--dataset", type=str, default="Vaihingen",
                            help="choose dataset. ['Potsdam', 'Vaihingen'].")
        parser.add_argument("--data_dir", type=str, default="./data",
                            help="Path to the directory containing the Cityscapes dataset.")
        parser.add_argument("--train_list", nargs='+', required=True,
                            help="Path to the train set listing the images in the dataset.")
        parser.add_argument("--test_list",  nargs='+', required=True,
                            help="Path to the test set listing the images in the dataset.")
        
        parser.add_argument("--model", type=str, default='U_Net', help="model architecture")
        parser.add_argument("--backbone", type=str, default='resnet50', help="model architecture")

        parser.add_argument("--aux_classifier", type=str2bool, default=True, 
            help="use the aux classifier in network architecture.")
        parser.add_argument("--auxloss_weight", type=float, default=0.4,
                    help="the loss weight to combine ce loss and ohem or lovasz loss.")

        parser.add_argument("--learning_rate", type=float, default=1e-2,
                            help="Base learning rate for training with polynomial decay.")
        parser.add_argument("--momentum", type=float, default=0.9,
                            help="Momentum component of the optimiser.")
        parser.add_argument("--num_classes", type=int, default=6,
                            help="Number of classes to predict (including background).")
        parser.add_argument("--max_epoches", type=int, default=200,
                            help="Epoches of training model.")
        parser.add_argument("--power", type=float, default=0.9,
                            help="Decay parameter to compute the learning rate.")
        parser.add_argument("--weight_decay", type=float, default=5e-4,
                            help="Regularisation parameter for L2-loss.")
        parser.add_argument("--num_workers", type=int, default=8)

        parser.add_argument('--lr-scheduler', type=str, default='poly',
                            help='learning rate scheduler (default: poly)')
        parser.add_argument('--warmup_epochs', type=int, default=0,
                            help='learning rate scheduler warmup setting!')

        # ***** Params for save and load ******
        parser.add_argument("--save_dir", type=str, default='./snapshots',
                            help="Where to save snapshots of the models.")
        parser.add_argument('--eval-interval', type=int, default=5,
                            help='evaluuation interval (default: 1)')
        parser.add_argument('--no-val', type=str2bool, default=False,
                            help='skip validation during training')

        args = parser.parse_args()
        return args

    start = time.time()
    args = get_arguments()

    # SEED
    SEED = 15
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    cudnn.benchmark = False
    cudnn.deterministic = True

    Seg = SegmentTrainer(args,definetime=None)
    testing_results_file = os.path.join(Seg.checkpoint_dir, 'testing_results_file.txt')
    # Seg.train(args)
    Seg.train(args)
    runtime = time.time() - start
    print(f"Spend Time: {math.floor(runtime//3600):2d}h:"
    f"{math.floor(runtime%3600//60):2d}m:{math.floor(runtime%60):2d}s")
    print(f"Model save in {Seg.exp_dir}")
    print(Seg.best_metrics)

    with open(testing_results_file, 'a') as f:
        f.write(f"Best_Epo: {Seg.best_metrics['epoch']}   Acc: {Seg.best_metrics['Acc']:.5f}  mIoU: {Seg.best_metrics['mIoU']:.5f}   mF1: {Seg.best_metrics['mF1']:.5f}"
               f"\nCategory F1: {Seg.best_metrics['F1']}\n")
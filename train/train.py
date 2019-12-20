#!/usr/bin/pytho
# -*- coding: UTF-8 -*-

import numpy as np
import os
import sys
# import dependencies
import time
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

#Pytorch requirements
import unicodedata
import string
import re
import random
import argparse

import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

#Fix import issues
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_dir)
os.chdir(project_dir)

import experiment
import config
import models.siamese as siamese
import models.base_model as base_model
from data_generator import Generator, classification_dataloader
from Logger import Logger
parser = argparse.ArgumentParser()

###############################################################################
#                             General Settings                                #
###############################################################################

parser.add_argument('--num_examples_train', nargs='?', const=1, type=int,
                    default=int(20000))
parser.add_argument('--num_examples_test', nargs='?', const=1, type=int,
                    default=int(1000))
parser.add_argument('--edge_density', nargs='?', const=1, type=float,
                    default=0.2)
parser.add_argument('--n_vertices', nargs='?', const=1, type=int, default=50)
parser.add_argument('--random_noise', action='store_true')
parser.add_argument('--noise', nargs='?', const=1, type=float, default=0.03)
parser.add_argument('--noise_model', nargs='?', const=1, type=int, default=2)
parser.add_argument('--generative_model', nargs='?', const=1, type=str,
                    default='ErdosRenyi')
parser.add_argument('--epoch', nargs='?', const=1, type=int,
                    default=5)
parser.add_argument('--batch_size', nargs='?', const=1, type=int, default=32)
parser.add_argument('--lr', nargs='?', const=1, type=float, default=1e-3)
parser.add_argument('--mode', nargs='?', const=1, type=str, default='train')
parser.add_argument('--path_dataset', nargs='?', const=1, type=str, default='dataset')
parser.add_argument('--path_logger', nargs='?', const=1, type=str, default='logger')
parser.add_argument('--path_exp', nargs='?', const=1, type=str, default='experiments/logs.json')
parser.add_argument('--print_freq', nargs='?', const=1, type=int, default=100)
parser.add_argument('--test_freq', nargs='?', const=1, type=int, default=500)
parser.add_argument('--save_freq', nargs='?', const=1, type=int, default=2000)
parser.add_argument('--clip_grad_norm', nargs='?', const=1, type=float,
                    default=40.0)

###############################################################################
#                                 GNN Settings                                #
###############################################################################

parser.add_argument('--num_features', nargs='?', const=1, type=int,
                    default=20)
parser.add_argument('--expressive_suffix', nargs='?', const=1, type=bool,
                    default=False)
parser.add_argument('--classification', nargs= '?', const=1, type=bool,
                    default=False)
parser.add_argument('--overfit', nargs= '?', const=1, type=bool,
                    default=False)
parser.add_argument('--num_layers', nargs='?', const=1, type=int,
                    default=20)
parser.add_argument('--num_blocks', nargs='?', const=1, type=int,
                    default=3)
parser.add_argument('--J', nargs='?', const=1, type=int, default=4)

args = parser.parse_args()

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    dtype_l = torch.cuda.LongTensor
    torch.cuda.manual_seed(0)
else:
    dtype = torch.FloatTensor
    dtype_l = torch.LongTensor
    torch.manual_seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def make_config(args):
    arch = {
            'depth_of_mlp' : args.num_layers,
            'block_features' : [args.num_features] * args.num_blocks,
            'expressive_suffix' : args.expressive_suffix,
            }
    conf = {
            'node_labels' : 1,
            'classification': args.classification,
            }
    return config.Configuration(conf, arch)


batch_size = args.batch_size
criterion = nn.CrossEntropyLoss()
template1 = '{:<10} {:<10} {:<10} {:<10} {:<15} {:<10} {:<10} {:<10} '
template2 = '{:<10} {:<10} {:<10.5f} {:<10.5f} {:<15} {:<10} {:<10} {:<10.3f} \n'


def compute_loss(pred, labels):
    pred = pred.view(-1, pred.size()[-1])
    labels = torch.flatten(labels)
    labels = labels.view(-1)
    return criterion(pred, labels)

def train(siamese_gnn, logger, gen, lr):
    siamese_gnn.train()
    labels = torch.arange(0, gen.n_vertices).unsqueeze(0).expand(batch_size, gen.n_vertices).to(device)
    optimizer = torch.optim.Adamax(siamese_gnn.parameters(), lr=lr)
    dataloader = gen.train_loader(args.batch_size)
    start = time.time()
    for epoch in range(args.epoch):
        for it, sample in enumerate(dataloader):
            sample = sample.to(device)
            pred = siamese_gnn(sample[:, 0], sample[:, 1])
            loss = compute_loss(pred, labels[: len(pred)])
            siamese_gnn.zero_grad()
            loss.backward()
            #nn.utils.clip_grad_norm(siamese_gnn.parameters(), args.clip_grad_norm)
            optimizer.step()
            logger.add_train_loss(loss)
            logger.add_train_accuracy(pred, labels[: len(pred)])
            elapsed = time.time() - start
            if it % logger.args['print_freq'] == 0:
                logger.plot_train_accuracy()
                logger.plot_train_loss()
                loss = loss.data.cpu().numpy()#[0]
                info = ['epoch', 'iteration', 'loss', 'accuracy', 'edge_density',
                    'noise', 'model', 'elapsed']
                out = [epoch, it, loss.item(), logger.accuracy_train[-1].item(), args.edge_density,
                   args.noise, args.generative_model, elapsed]
                print(template1.format(*info))
                print(template2.format(*out))
            if it % logger.args['save_freq'] == 0:
                logger.save_model(siamese_gnn)
                logger.save_results()
    print('Optimization finished.')
    logger.save_model(siamese_gnn)

def test(siamese_gnn, logger, gen):
    siamese_gnn.eval()
    labels = torch.arange(0, gen.n_vertices).unsqueeze(0).expand(batch_size, gen.n_vertices).to(device)
    dataloader = gen.test_loader(args.batch_size)
    for it, sample in enumerate(dataloader):
        sample = sample.to(device)
        pred = siamese_gnn(sample[:, 0], sample[:, 1])
        logger.add_test_accuracy(pred, labels[: len(pred)])
    accuracy = sum(logger.accuracy_test)/len(logger.accuracy_test)
    print('Accuracy: ', accuracy)
    return accuracy

def setup():
    logger = Logger(args.path_logger)
    logger.write_settings(args)
    config = make_config(args)
    model = base_model.BaseModel(config)
    siamese_gnn = siamese.Siamese(model).to(device)
    gen = Generator()
    gen.set_args(vars(args))
    gen.load_dataset()
    return siamese_gnn, logger, gen

def make_qap():
    siamese_gnn, logger, gen = setup()
    if args.mode == 'train':
        train(siamese_gnn, logger, gen, args.lr)
    if args.mode == 'test':
        siamese_gnn = logger.load_model()
        test(siamese_gnn, logger, gen)
    if args.mode == 'experiment':
        start = time.time()
        train(siamese_gnn, logger, gen, args.lr)
        siamese_gnn = logger.load_model()
        acc = test(siamese_gnn, logger, gen)
        end = time.time()
        delta = end - start
        experiment.save_experiment(args, acc, delta)

###############################################################################
#                                Classification                               #
###############################################################################

def classification_setup():
    logger = Logger(args.path_logger)
    logger.write_settings(args)
    config = make_config(args)
    model = base_model.BaseModel(config).to(device)
    dataloaders = classification_dataloader(args)
    return model, logger, dataloaders

def classification_train(model, logger, dataloader, lr):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    start = time.time()
    for epoch in range(args.epoch):
        for it, (sample, label) in enumerate(dataloader):
            model.zero_grad()
            sample = sample.to(device)
            label = label.to(device)
            pred = model(sample)
            loss = criterion(pred, label)
            #print('label: ', label)
            #print('pred: ', pred)
            loss.backward()
            optimizer.step()
            logger.add_train_loss(loss)
            accuracy = ((label - pred.max(-1)[1])**2).float().mean()
            elapsed = time.time() - start
            if it % logger.args['print_freq'] == 0:
                loss = loss.data.cpu().numpy()#[0]
                info = ['epoch', 'iteration', 'loss', 'accuracy', 'elapsed']
                out = [epoch, it, loss.item(), accuracy.item(), elapsed]
                print(("{:<10} "*5).format(*info))
                print(("{:<10} "*2 + "{:<10.5f} "*2 + "{:<10.3f}").format(*out))
        n, p = list(model.named_parameters())[-3]
        assert p.requires_grad
        print(n, p.data)
        print(p.grad.data)
    print('Optimization finished.')

def classification_test(model, logger, dataloader):
    model.eval()
    acc = 0
    for it, (sample, label) in enumerate(dataloader):
        sample = sample.to(device)
        label = label.to(device)
        pred = model(sample)
        accuracy = ((label - pred.max(-1)[1])**2).float().mean().item()
        acc += accuracy
    return acc/(it+1)

def make_classification():
    print(args)
    start = time.time()
    model, logger, dataloaders = classification_setup()
    classification_train(model, logger, dataloaders[0], args.lr)
    if args.overfit:
        acc = classification_test(model, logger, dataloaders[0])
    else:
        acc = classification_test(model, logger, dataloaders[1])
    end = time.time()
    print('accuracy: ', acc)
    delta = end - start
    experiment.save_experiment(args, acc, delta)


###############################################################################
#                                   Main                                      #
###############################################################################

if __name__ == '__main__':
    if args.classification:
        make_classification()
    else:
        make_qap()

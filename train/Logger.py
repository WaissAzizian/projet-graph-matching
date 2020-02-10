import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial import ConvexHull

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import lap

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    dtype_l = torch.cuda.LongTensor
    torch.cuda.manual_seed(0)
else:
    dtype = torch.FloatTensor
    dtype_l = torch.LongTensor
    torch.manual_seed(0)

def predict_lap(matrices):
    assert len(matrices.size()) == 3, matrices.size()
    np_matrices = matrices.data.cpu().numpy()
    permutations = np.empty((matrices.size(0), matrices.size(1)), dtype=int)
    for index, matrix in enumerate(np_matrices):
        assert len(matrix.shape) == 2, cost_matrix.shape
        permutation, _ = lap.lapjv(-matrix, return_cost=False)
        permutations[index] = permutation
    return permutations

def accuracy_lap(pred):
    permutations = predict_lap(pred)
    m = permutations.shape[1]
    identity = np.arange(m)
    acc = np.mean(permutations == identity[np.newaxis, :])
    return acc

def compute_recovery_rate(pred, labels, lap):
    if not lap:
        pred = pred.max(-1)[1]
        error = 1 - torch.eq(pred, labels).type(dtype)#.squeeze(2)
        frob_norm = error.mean(1)#.squeeze(1)
        accuracy = 1 - frob_norm
        accuracy = accuracy.mean(0).squeeze()
        return accuracy.data.cpu().numpy()
    else:
        return accuracy_lap(pred)

class Logger(object):
    def __init__(self, path_logger):
        directory = os.path.join(path_logger, 'plots/')
        self.path = path_logger
        self.path_dir = directory
        # Create directory if necessary
        try:
            os.stat(directory)
        except:
            os.mkdir(directory)
        self.loss_train = []
        self.loss_test = []
        self.accuracy_train_lap = []
        self.accuracy_train_plain = []
        self.accuracy_test_lap = []
        self.accuracy_test_plain = []
        self.args = None

    def write_settings(self, args):
        self.args = {}
        # write info
        path = os.path.join(self.path, 'experiment.txt')
        with open(path, 'w') as file:
            for arg in vars(args):
                file.write(str(arg) + ' : ' + str(getattr(args, arg)) + '\n')
                self.args[str(arg)] = getattr(args, arg)

    def save_model(self, model):
        save_dir = os.path.join(self.path, 'parameters/')
        # Create directory if necessary
        try:
            os.stat(save_dir)
        except:
            os.mkdir(save_dir)
        path = os.path.join(save_dir, 'gnn.pt')
        torch.save(model, path)
        print('Model Saved.')

    def load_model(self):
        load_dir = os.path.join(self.path, 'parameters/')
        # check if any training has been done before.
        try:
            os.stat(load_dir)
        except:
            print("Training has not been done before testing. This session will be terminated.")
            sys.exit()
        path = os.path.join(load_dir, 'gnn.pt')
        print('Loading the most recent model...')
        siamese_gnn = torch.load(path)
        return siamese_gnn

    def add_train_loss(self, loss):
        self.loss_train.append(loss.data.cpu().numpy())

    def add_test_loss(self, loss):
        self.loss_test.append(loss)

    def add_train_accuracy(self, pred, labels):
        accuracy_lap = compute_recovery_rate(pred, labels, True)
        self.accuracy_train_lap.append(accuracy_lap)
        accuracy_plain = compute_recovery_rate(pred, labels, False)
        self.accuracy_train_plain.append(accuracy_plain)

    def add_test_accuracy(self, pred, labels):
        accuracy_lap = compute_recovery_rate(pred, labels, True)
        self.accuracy_test_lap.append(accuracy_lap)
        accuracy_plain = compute_recovery_rate(pred, labels, False)
        self.accuracy_test_plain.append(accuracy_plain)

    def plot_train_loss(self):
        plt.figure(0)
        plt.clf()
        iters = range(len(self.loss_train))
        plt.semilogy(iters, self.loss_train, 'b')
        plt.xlabel('iterations')
        plt.ylabel('Cross Entropy Loss')
        plt.title('Training Loss: p={}, p_e={}'
                  .format(self.args['edge_density'], self.args['noise']))
        path = os.path.join(self.path_dir, 'training_loss.png')
        plt.savefig(path)

    def plot_test_loss(self):
        plt.figure(1)
        plt.clf()
        test_freq = self.args['test_freq']
        iters = test_freq * range(len(self.loss_test))
        plt.semilogy(iters, self.loss_test, 'b')
        plt.xlabel('iterations')
        plt.ylabel('Cross Entropy Loss')
        plt.title('Testing Loss: p={}, p_e={}'
                  .format(self.args['edge_density'], self.args['noise']))
        path = os.path.join(self.path_dir, 'testing_loss.png')
        plt.savefig(path)

    def plot_train_accuracy(self):
        plt.figure(0)
        plt.clf()
        iters = range(len(self.accuracy_train))
        plt.plot(iters, self.accuracy_train, 'b')
        plt.xlabel('iterations')
        plt.ylabel('Accuracy')
        plt.title('Training Accuracy: p={}, p_e={}'
                  .format(self.args['edge_density'], self.args['noise']))
        path = os.path.join(self.path_dir, 'training_accuracy.png')
        plt.savefig(path)

    def plot_test_accuracy(self):
        plt.figure(1)
        plt.clf()
        test_freq = self.args['test_freq']
        iters = test_freq * range(len(self.accuracy_test))
        plt.plot(iters, self.accuracy_test, 'b')
        plt.xlabel('iterations')
        plt.ylabel('Accuracy')
        plt.title('Testing Accuracy: p={}, p_e={}'
                  .format(self.args['edge_density'], self.args['noise']))
        path = os.path.join(self.path_dir, 'testing_accuracy.png')
        plt.savefig(path)

    def save_results(self):
        path = os.path.join(self.path, 'results.npz')
        np.savez(path, accuracy_train=np.array(self.accuracy_train_lap),
                 accuracy_test=np.array(self.accuracy_test_lap),
                 loss_train=self.loss_train, loss_test=self.loss_test)

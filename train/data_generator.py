import numpy as np
import os
import networkx
import torch
import torch.nn as nn
import torch.utils

class Generator(object):
    def __init__(self, path_dataset):
        self.path_dataset = path_dataset
        self.num_examples_train = 10e6
        self.num_examples_test = 10e4
        self.data_train = []
        self.data_test = []
        self.n_vertices = 50 #number vertices
        self.generative_model = 'ErdosRenyi'
        self.edge_density = 0.2
        self.random_noise = False
        self.noise = 0.03
        self.noise_model = 2

    def ErdosRenyi(self, p, N):
        W = np.zeros((N, N))
        for i in range(0, N - 1):
            for j in range(i + 1, N):
                add_edge = (np.random.uniform(0, 1) < p)
                if add_edge:
                    W[i, j] = 1
                W[j, i] = W[i, j]
        return W

    def ErdosRenyi_netx(self, p, N):
        g = networkx.erdos_renyi_graph(N, p)
        W = networkx.adjacency_matrix(g).todense().astype(float)
        W = np.array(W)
        return W

    def RegularGraph_netx(self, p, N):
        """ Generate random regular graph """
        d = p * N
        d = int(d)
        g = networkx.random_regular_graph(d, N)
        W = networkx.adjacency_matrix(g).todense().astype(float)
        W = np.array(W)
        return W

    def adjacency_matrix_to_tensor_representation(self, W):
        degrees = W.sum(1)
        B = np.zeros((self.n_vertices, self.n_vertices, 2))
        B[:, :, 1] = W
        indices = np.arange(self.n_vertices)
        B[indices, indices, 0] = degrees
        return B

    def compute_example(self):
        if self.generative_model == 'ErdosRenyi':
            W = self.ErdosRenyi_netx(self.edge_density, self.n_vertices)
        elif self.generative_model == 'Regular':
            W = self.RegularGraph_netx(self.edge_density, self.n_vertices)
        else:
            raise ValueError('Generative model {} not supported'
                             .format(self.generative_model))
        if self.random_noise:
            self.noise = np.random.uniform(0.000, 0.050, 1)
        if self.noise_model == 1:
            # use noise model from [arxiv 1602.04181], eq (3.8)
            noise = self.ErdosRenyi(self.noise, self.n_vertices)
            W_noise = W*(1-noise) + (1-W)*noise
        elif self.noise_model == 2:
            # use noise model from [arxiv 1602.04181], eq (3.9)
            pe1 = self.noise
            pe2 = (self.edge_density*self.noise)/(1.0-self.edge_density)
            noise1 = self.ErdosRenyi_netx(pe1, self.n_vertices)
            noise2 = self.ErdosRenyi_netx(pe2, self.n_vertices)
            W_noise = W*(1-noise1) + (1-W)*noise2
        else:
            raise ValueError('Noise model {} not implemented'
                             .format(self.noise_model))
        B = self.adjacency_matrix_to_tensor_representation(W)
        B_noise = self.adjacency_matrix_to_tensor_representation(W_noise)
        return (B, B_noise)
    
    def create_dataset_test(self):
        for i in range(self.num_examples_test):
            example = self.compute_example()
            self.data_test.append(example)
    
    def create_dataset_train(self):
        for i in range(self.num_examples_train):
            example = self.compute_example()
            self.data_train.append(example)

    def load_dataset(self):
        # load train dataset
        if self.random_noise:
            filename = 'QAPtrain_RN.np'
        else:
            filename = ('QAPtrain_{}_{}_{}.np'.format(self.generative_model,
                        self.noise, self.edge_density))
        path = os.path.join(self.path_dataset, filename)
        if os.path.exists(path):
            print('Reading training dataset at {}'.format(path))
            with open(path, 'rb') as f:
                self.data_train = np.load(f, allow_pickle=True)
        else:
            print('Creating training dataset.')
            self.create_dataset_train()
            print('Saving training datatset at {}'.format(path))
            with open(path, 'wb') as f:
                np.save(f, self.data_train)
        # load test dataset
        if self.random_noise:
            filename = 'QAPtest_RN.np'
        else:
            filename = ('QAPtest_{}_{}_{}.np'.format(self.generative_model,
                        self.noise, self.edge_density))
        path = os.path.join(self.path_dataset, filename)
        if os.path.exists(path):
            print('Reading testing dataset at {}'.format(path))
            with open(path, 'rb') as f:
                self.data_test = np.load(f, allow_pickle=True)
        else:
            print('Creating testing dataset.')
            self.create_dataset_test()
            print('Saving testing datatset at {}'.format(path))
            with open(path, 'wb') as f:
                np.save(f, self.data_test)

    def clean_datasets(self):
        for usage in ['train', 'test']:
            if self.random_noise: 
                filename = 'QAP{}_RN.np'.format(usage)
            else:
                filename = ('QAP{}_{}_{}_{}.np'.format(usage, self.generative_model,
                        self.noise, self.edge_density))
            path = os.path.join(self.path_dataset, filename)
            if os.path.exists(path):
                os.remove(path)


    def train_loader(self, batch_size):
        assert len(self.data_train) > 0
        torch_data_train = torch.Tensor(self.data_train)
        return torch.utils.data.DataLoader(torch_data_train, batch_size=batch_size, shuffle=True, num_workers=4)
    
    def test_loader(self, batch_size):
        assert len(self.data_test) > 0
        torch_data_test = torch.Tensor(self.data_test)
        return torch.utils.data.DataLoader(torch_data_test, batch_size=batch_size, shuffle=True, num_workers=4)





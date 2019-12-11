import torch
import torch.nn as nn
import unittest
import os
import sys

#Fix import issues
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_dir)
os.chdir(project_dir)

#Project imports
import base_model

#repeat decorator
def repeat(n):
    def dec(f):
        def new_func(*args):
            for _ in range(n):
                f(*args)
        return new_func
    return dec


#Dummy configuration for testing
class Architecture:
    def __init__(self, input_features, block_features, depth_of_mlp, expressive_suffix):
        self.block_features = block_features
        self.input_features = input_features
        self.depth_of_mlp = depth_of_mlp
        self.expressive_suffix = expressive_suffix

class Configuration:
    def __init__(self, node_labels, block_features, depth_of_mlp, expressive_suffix):
        self.node_labels = node_labels
        input_features = node_labels + 1
        self.architecture = Architecture(input_features, block_features, depth_of_mlp, expressive_suffix)

class TestPPGNNModel(unittest.TestCase):
    def setUp(self):
        self.configs = Configuration(100, [10, 10], 2, False), Configuration(100, [10, 10], 2, True)
        self.batch_size = 64
        self.n_vertices = 16

    def test_dimensions(self):
        for config in self.configs:
            x = torch.zeros(self.batch_size, self.n_vertices, self.n_vertices, config.architecture.input_features)
            model = base_model.BaseModel(config)
            self.assertEqual(model(x).size(), (self.batch_size, self.n_vertices, config.architecture.block_features[-1]))

    @repeat(10)
    def test_equivariance(self):
        for config in self.configs:
            x = torch.zeros(self.batch_size, self.n_vertices, self.n_vertices, config.architecture.input_features)
            x.normal_()
            model = base_model.BaseModel(config)
            output1 = model(x)
            perm = torch.randperm(self.n_vertices)
            x_perm = x[:, perm][:,:,  perm]
            output2 = model(x_perm)
            output_perm = output1[:, perm]
            self.assertTrue(torch.allclose(output_perm, output2, atol=1e-5))


if __name__ == '__main__':
    unittest.main()

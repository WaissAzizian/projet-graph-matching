import unittest
import data_generator

class Test(unittest.TestCase):
    def test_dim(self):
        gen = data_generator.Generator('dataset')
        gen.num_examples_train = 5
        gen.num_examples_test = 5
        gen.load_dataset()
        batch_size = 3
        loader = gen.train_loader(batch_size)
        batch = next(iter(loader))
        self.assertEqual(batch.size(), (batch_size, 2, gen.n_vertices, gen.n_vertices, 2))
        gen.clean_datasets()

if __name__ == '__main__':
    unittest.main()

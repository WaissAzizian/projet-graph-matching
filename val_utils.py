#!/usr/bin/env python

import os
import torch
import argparse

def get_best(directory, display=['lr', 'gamma', 'num_features', 'num_layers']):
    best_acc, acc = 0, 0
    best_param, param = None, None
    best_filename = None
    for filename in os.listdir(directory):
        if filename.endswith('.pkl'):
            with open(os.path.join(directory, filename), 'rb') as f:
                param = torch.load(f)
                acc = torch.load(f)
                lst = ['{}={:.5f}'.format(p, vars(param)[p]) for p in display]
                s = ', '.join(lst)
                print('[{}] {:.3f} ({})'.format(s, acc, filename)) 
                if acc > best_acc:
                    best_acc = acc
                    best_param = param
                    best_filename = filename
    return best_filename, best_param, best_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", help="directory where the pickle files are stored", default="./val_results")
    args = parser.parse_args()
    f, param, acc = get_best(args.directory)
    print('Best accuracy: {:.3f}'.format(acc))
    print('File: {}'.format(f))
    print('Parameters: {}'.format(param))


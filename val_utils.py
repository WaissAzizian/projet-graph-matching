#!/usr/bin/env python

import os
import torch

def get_best(dir):
    best_acc, acc = None, None
    best_param, param = None, None
    best_filename = None
    for filename in os.listdir(dir):
        if filename.endswith('.pkl'):
            with open(filename, 'rb') as f:
                param = torch.load(f)
                acc = torch.load(f)
                if acc > best_acc:
                    best_acc = acc
                    best_param = param
                    best_filename = filename
    return best_filename, best_param, best_acc

if __name__ == '__main__':
    f, param, acc = get_best('./val_results')
    print('Best accuracy: {}'.format(acc))
    print('File: {}'.format(f))
    print('Parameters: {}'.format(f))


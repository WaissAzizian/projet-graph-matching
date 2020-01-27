#!/usr/bin/env python

import torch
import geomloss
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--blur"  , type=float, default=0.05)
parser.add_argument("--scaling" , type=float, default= 0.5)
args = parser.parse_args()

otloss = geomloss.SamplesLoss(blur=args.blur, scaling=args.scaling)
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
if __name__ == '__main__':
    with open('wrong_data.pkl', 'rb') as f:
        x = torch.load(f, map_location=device).to(device)
        y = torch.load(f, map_location=device).to(device)
        print(otloss(x, y).cpu().data[12])
        # print([otloss(a, b).item() for (a, b) in zip(x, y)])
        print(otloss(x[12], y[12]))
        dist = torch.cdist(x[12], y[12])
        print(torch.min(dist), torch.max(dist))
    with open('wrong_data_item.pkl', 'wb') as f:
        torch.save(x[12], f)
        torch.save(y[12], f)
        

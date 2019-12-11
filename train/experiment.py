import os
import datetime
import json

def save_experiment(args, acc):
    git_commit = os.popen('git describe --always').read()
    d = vars(args)
    d['commit'] = git_commit.replace("\n", "")
    now = datetime.datetime.now()
    d['end'] = now.strftime("%d/%m/%Y %H:%M:%S")
    d['accuracy'] = acc
    with open(args.path_exp, 'a') as f:
        json.dump(d, f)
        f.write("\n")

import os
import datetime
import json
import time

def save_experiment(args, acc, elapsed):
    git_commit = os.popen('git describe --always').read()
    d = vars(args)
    d['commit'] = git_commit.replace("\n", "")
    now = datetime.datetime.now()
    d['end'] = now.strftime("%d/%m/%Y %H:%M:%S")
    d['accuracy'] = acc
    d['elapsed'] = time.strftime("%H:%M:%S", time.gmtime(elapsed))
    with open(args.path_exp, 'a') as f:
        json.dump(d, f)
        f.write("\n")

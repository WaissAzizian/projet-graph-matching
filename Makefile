.PHONY: unittests
unittests: clean
	python3 models/utests.py
	python3 train/utests.py

quick_train:
	python3 train/train.py --num_examples_train 1000 --print_freq 100 --epoch 2

quick_test:
	python3 train/train.py --num_examples_test 100 --mode test

quick_experiment:
	python3 train/train.py --expressive_suffix True --num_examples_train 10 --num_examples_test 10 --print_freq 100 --epoch 1 --mode experiment --lr 0.0001

overfit:
	python3 train/train.py --expressive_suffix False --num_examples_train 5 --num_examples_test 995 --print_freq 1000 --epoch 50 --lr 0.01 --overfit --classification --num_layers 2 --num_features 2 --num_blocks 1

experiments:
	#python3 train/train.py --expressive_suffix True  --num_examples_train 20000 --num_examples_test 1000 --print_freq 1000 --epoch 5 --mode experiment
	python3 train/train.py --expressive_suffix False --num_examples_train 20000 --num_examples_test 1000 --print_freq 1000 --epoch 5 --mode experiment --num_features 100
	python3 train/train.py --expressive_suffix False --num_examples_train 20000 --num_examples_test 1000 --print_freq 1000 --epoch 5 --mode experiment --num_layers 50
	python3 train/train.py --expressive_suffix False --num_examples_train 20000 --num_examples_test 1000 --print_freq 1000 --epoch 5 --mode experiment --num_blocks 6
	python3 train/train.py --expressive_suffix False --num_examples_train 20000 --num_examples_test 1000 --print_freq 1000 --epoch 10 --mode experiment
	python3 train/train.py --expressive_suffix False --num_examples_train 20000 --num_examples_test 1000 --print_freq 1000 --epoch 5 --mode experiment --lr 0.0001

clean:
	rm -rf dataset/*

quick_classification:
	python3 train/train.py --num_examples_train 800 --num_examples_test 200 --classification --print_freq 100 --epoch 5 --num_features 2 --num_layers 2 --num_blocks 1 --lr 0.01

classification_experiment:
	python3 train/train.py --num_examples_train 800 --num_examples_test 200 --classification --epoch 50 --num_features 2 --num_blocks 3 --num_layers 10 --lr 0.001

INSTANCE=pytorch-instance-p4
PROJECT=homework-nlp
ZONE=us-west2-c
HOST=$(INSTANCE).$(ZONE).$(PROJECT)

start:
	gcloud compute instances start $(INSTANCE)
	gcloud compute config-ssh

stop:
	gcloud compute instances stop $(INSTANCE)

fetch_results:
	rsync -Pavu "$(HOST)":/home/$(GUSER)/projet-graph-matching/experiments/ "$(PWD)"/experiments/

connect:
	ssh $(HOST)

list:
	gcloud compute instances list

pull_git:
	git fetch --all
	git reset --hard origin/master

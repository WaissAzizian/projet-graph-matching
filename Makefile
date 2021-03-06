.PHONY: unittests
unittests: clean
	python3 models/utests.py
	python3 train/utests.py

quick_train:
	python3 train/train.py --num_examples_train 1000 --print_freq 100 --epoch 2

quick_test:
	python3 train/train.py --num_examples_test 100 --mode test

quick_experiment:
	python3 train/train.py --num_examples_train 10 --num_examples_test 10 --num_examples_val 2230 --n_vertices 50 --print_freq 100 --epoch 50 --mode experiment --lr 0.001 --num_layers 3 --num_blocks 2 --num_features 16

CMD="python3 train/train.py --num_examples_train 20000 --num_examples_test 1000 --mode validation --print_freq 1000 --epoch 100 --step_epoch 5 --num_blocks 2 --generative_model Regular --noise 0.05"
validation:
	python3 -m grid val_results_reg_5 $(CMD) --lr 1e-3 5e-3 1e-2 --gamma 0.9 0.99 0.999 --num_features 16 32 64 --num_layers 3 5 --n 3

overfit:
	python3 train/train.py --expressive_suffix False --num_examples_train 5 --num_examples_test 995 --print_freq 1000 --epoch 50 --lr 0.01 --gamma 0.75 --overfit --classification --num_layers 2 --num_features 2 --num_blocks 1

experiment:
	python3 train/train.py --num_examples_train 750 --num_examples_val 1 --num_examples_test 249 --n_vertices 136 --print_freq 100 --epoch 50 --mode experiment --lr 0.001 --num_layers 3 --num_blocks 2 --num_features 32 --real_world_dataset --gamma 0.9

pretrained_classification:
	python3 train/train.py --num_examples_train 750 --num_examples_val 249 --num_examples_test 1 --pretrained_classification --batch_size 8
clean:
	rm -rf dataset/*

quick_classification:
	python3 train/train.py --num_examples_train 800 --num_examples_test 200 --classification --print_freq 100 --epoch 5 --num_features 2 --num_layers 2 --num_blocks 1 --lr 0.01

classification_experiment:
	python3 train/train.py --num_examples_train 800 --num_examples_test 200 --classification --print_freq 10 --epoch 100 --num_features 2 --num_layers 3 --num_blocks 2 --lr 0.05 --gamma 0.7

CCMD="python3 train/train.py --num_examples_train 700 --num_examples_test 150 --num_examples_val 150 --classification --validation --print_freq 10 --epoch 200 --num_features 2 --num_layers 3 --num_blocks 2"
classification_val:
	python3 -m grid val_results $(CCMD) --lr 5e-5 1e-4 5e-4 1e-3 --gamma 0.5 0.75 1 --n 3


INSTANCE=pytorch-instance-p4
PROJECT=homework-nlp
ZONE=us-west2-c
HOST=$(INSTANCE).$(ZONE).$(PROJECT)

start:
	gcloud compute instances start $(INSTANCE)

stop:
	gcloud compute instances stop $(INSTANCE)

fetch_results:
	rsync -Pavu "$(HOST)":/home/$(GUSER)/projet-graph-matching/experiments/ "$(PWD)"/experiments/

connect:
	gcloud compute config-ssh
	ssh $(HOST)

list:
	gcloud compute instances list

pull_git:
	git fetch --all
	git reset --hard origin/master

requirements:
	pip install -r requirements.txt --user

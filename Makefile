.PHONY: unittests
unittests: clean
	python3 models/utests.py
	python3 train/utests.py

quick_train:
	python3 train/train.py --num_examples_train 1000 --print_freq 100 --epoch 2

quick_test:
	python3 train/train.py --num_examples_test 100 --mode test

quick_experiment:
	python3 train/train.py --expressive_suffix True --num_examples_train 10 --num_examples_test 10 --print_freq 100 --epoch 1 --mode experiment

make_experiments:
	python3 train/train.py --expressive_suffix True  --num_examples_train 20000 --num_examples_test 1000 --print_freq 1000 --epoch 5 --mode experiment
	python3 train/train.py --expressive_suffix False --num_examples_train 20000 --num_examples_test 1000 --print_freq 1000 --epoch 5 --mode experiment

clean:
	rm -rf dataset/*

INSTANCE=pytorch-instance-p4
PROJECT=homework-nlp
ZONE=us-west2-c
HOST=$(INSTANCE).$(ZONE).$(PROJECT)

start:
	gcloud compute instances start $(INSTANCE)
	gcloud compute config-ssh

stop:
	gcloud compute instances stop $(INSTANCE)

deploy:
	rsync ssh -Pavur "$(PWD)" "$(HOST)":/home/waissfowl/

connect:
	ssh $(HOST)

list:
	gcloud compute instances list


.PHONY: unittests
unittests: clean
	python3 models/utests.py
	python3 train/utests.py

quick_train:
	python3 train/train.py --num_examples_train 1000 --print_freq 100 --epoch 2

quick_test:
	python3 train/train.py --num_examples_test 100 --mode test

quick_experiment:
	python3 train/train.py --expressive_suffix True --num_examples_train 100 --num_examples_test 100 --print_freq 100 --epoch 2 --mode experiment

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

deploy: test
	rsync ssh -Pavur "$(PWD)" "$(HOST)":/home/waissfowl/

connect:
	ssh $(HOST)

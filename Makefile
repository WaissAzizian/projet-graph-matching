.PHONY: unittests
unittests: clean
	python3 models/utests.py
	python3 train/utests.py

quick_train:
	python3 train/train.py --num_examples_train 12 --print_freq 10 --num_examples_test 10 --epoch 2
	python3 train/train.py --num_examples_train 12 --print_freq 10 --num_examples_test 10 --epoch 2 --mode test

clean:
	rm -rf dataset/*

.PHONY: unittests
unittests: clean
	python3 models/utests.py
	python3 train/utests.py

clean:
	rm -rf dataset/*

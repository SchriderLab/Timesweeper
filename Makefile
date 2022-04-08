.PHONY: env install

all: env build_slim

env:
	conda env create -f blinx.yml

build_slim:
	wget http://benhaller.com/slim/SLiM.zip
	unzip SLiM.zip
	rm SLiM.zip
	mkdir SLiM/build
	cd SLiM/build; cmake ..
	cd SLiM/build; make
	mv SLiM src/SLiM

install: build_slim

build:
	rm src/timesweeper/timesweeper.egg-info/ -rf
	python -m build
	pip install .
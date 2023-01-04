.PHONY: env

all: env build

env:
	conda env create -f blinx.yml

build:
	rm src/timesweeper/timesweeper.egg-info/ -rf
	python -m build
	conda activate blinx; pip install -e .

sf:
	wget http://degiorgiogroup.fau.edu/SF2.tar.gz
	tar -xzvf SF2.tar.gz
	cd SF2/; make
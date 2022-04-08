.PHONY: env

all: env build

env:
	conda env create -f blinx.yml

build:
	rm src/timesweeper/timesweeper.egg-info/ -rf
	python -m build
	conda activate blinx; pip install -e .
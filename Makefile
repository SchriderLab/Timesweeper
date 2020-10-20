.PHONY: clean sims combine format train plot

clean:
	rm */*/*/*/*log
	rm */*/*/*/*npz

# Utilities
environment:
	conda env create -f timesweeper.yml
	
freeze:
	conda env export > timesweeper.yml

slim:
	wget http://benhaller.com/slim/SLiM.zip
	unzip SLiM.zip
	mkdir SLiM/build
	cd SLiM/build
	cmake ../SLiM
	make slim

exppath:
	cd build
	export PATH="$PWD:$PATH"

# Running Python 
sims:
	python timeseriessweeps/tss.py -f launch_sims

combine:
	python timeseriessweeps/tss.py -f combine_sims

format:
	python timeseriessweeps/tss.py -f format_all

train:
	python timeseriessweeps/tss.py -f train_nets

plot:
	python timeseriessweeps/tss.py -f plot_input_npz
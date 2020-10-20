.PHONY: clean sims combine format train plot

clean:
	rm */*/*/*log
	rm */*/*/*npz

# Utilities
env:
	echo "\nTake your time...\n"
	conda env create -f blinx.yml
	
freeze:
	conda env export > blinx.yml

slim:
	rm -rf CMakeFiles SLiM
	wget http://benhaller.com/slim/SLiM.zip
	unzip SLiM.zip
	rm SLiM.zip
	mkdir SLiM/build
	cd SLiM/build; cmake ..
	cd SLiM/build; make

shic:
	rm -rf diploSHIC
	git clone https://github.com/kern-lab/diploSHIC.git
	source activate blinx; cd diploSHIC; python setup.py install

install: env slim shic
	echo "\nKept ya waiting, huh?\n"

# Running Python 
sims:
	python timesweeper/blinx.py -f launch_sims

combine:
	python timesweeper/blinx.py -f combine_sims

format:
	python timesweeper/blinx.py -f format_all

train:
	python timesweeper/blinx.py -f train_nets

plot:
	python timesweeper/blinx.py -f plot_input_npz

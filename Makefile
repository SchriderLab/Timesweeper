.PHONY: cleanup sims wash train plot

cleanup:
	rm -f */**/*log
	rm -rf test/sims/* 
	rm -f */**.txt
	rm -rf */simDumps
	rm -f */**.fvec


# Utilities
env:
	echo "\nTake your time...\n"
	conda env create -f blinx.yml
	
freeze:
	conda env export > blinx.yml

build_slim:
	rm -rf CMakeFiles SLiM
	wget http://benhaller.com/slim/SLiM.zip
	unzip SLiM.zip
	rm SLiM.zip
	mkdir SLiM/build
	cd SLiM/build; cmake ..
	cd SLiM/build; make

build_shic:
	rm -rf diploSHIC
	git clone https://github.com/kern-lab/diploSHIC.git
	source activate blinx; cd diploSHIC; python setup.py install

install: env slim shic
	echo "\nKept ya waiting, huh?\n"


# Running Python sub-modules
sims:
	python timesweeper/blinx.py -f launch_sims

wash:
	python timesweeper/blinx.py -f clean_sims

shic:
	python timesweeper/blinx.py -f create_feat_vecs

train:
	python timesweeper/blinx.py -f train_nets


#Testing
onesim:
	python /proj/dschridelab/timeSeriesSweeps/timesweeper/runAndParseSlim.py /proj/dschridelab/timeSeriesSweeps/test/test.slim 20 4 100 40 1 200 100 100000 True hard /proj/dschridelab/timeSeriesSweeps/test/simDumps/hard/hard_1.trees.dump > /proj/dschridelab/timeSeriesSweeps/test/sims/hard/hard_1.msOut

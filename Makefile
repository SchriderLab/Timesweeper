.PHONY: cleanup env freeze build_slim build_shic install

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

install: build_slim build_shic
	echo "Kept ya waiting, huh?"


#Testing
onesim:
	python /proj/dschridelab/timeSeriesSweeps/timesweeper/runAndParseSlim.py /proj/dschridelab/timeSeriesSweeps/test/test.slim 20 4 100 40 1 200 100 100000 True hard /proj/dschridelab/timeSeriesSweeps/test/simDumps/hard/hard_1.trees.dump > /proj/dschridelab/timeSeriesSweeps/test/sims/hard/hard_1.msOut

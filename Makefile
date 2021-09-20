.PHONY: cleanup env freeze build_slim build_shic install

cleanup:
	rm -f */**/*log
	rm -rf test/sims/* 
	rm -f */**.txt
	rm -rf */simDumps
	rm -f */**.fvec


# Utilities
env:
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

msmc_tools:
	mkdir mongolian_model/; cd mongolian_model; git clone https://github.com/stschiff/msmc-tools.git

install: build_slim msmc_tools
	echo "Kept ya waiting, huh?"


#Testing
onesim:
	python /proj/dschridelab/timeSeriesSweeps/src/runAndParseSlim.py /proj/dschridelab/timeSeriesSweeps/test/test.slim 20 4 100 40 1 200 100 100000 True hard /proj/dschridelab/timeSeriesSweeps/test/simDumps/hard/hard_1.trees.dump > /proj/dschridelab/timeSeriesSweeps/test/sims/hard/hard_1.msOut

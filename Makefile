.PHONY: cleanup env freeze build_slim build_shic install

cleanup:

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

install: build_slim msmc_tools

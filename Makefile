.PHONY: clean

clean:
	rm */*/*/*/*log
	rm */*/*/*/*npz

environment:
	conda env create -f timesweeper.yml
	
freeze:
	conda env export > timesweeper.yml

slim:
	wget http://benhaller.com/slim/SLiM.zip
	unzip SLiM.zip
	mkdir build
	cd build
	cmake ../SLiM
	make slim

exppath:
	cd build
	export PATH="$PWD:$PATH"

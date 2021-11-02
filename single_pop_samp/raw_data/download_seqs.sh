module load sratoolkit
fasterq-dump --split-files $(cat accessions.txt)
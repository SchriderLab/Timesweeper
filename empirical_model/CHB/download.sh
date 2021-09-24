module load sratoolkit

#for i in $(cat accessions.txt)
#    do
#        prefetch -X 40G $i
#    done
conda activate blinx
for i in $(cat accessions.txt)
    do
    	parallel-fastq-dump --sra-id $i --threads 8 --split-files
    done

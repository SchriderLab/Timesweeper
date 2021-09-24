module load sratoolkit

for i in $(cat accessions.txt)
    do
        prefetch -X 40G $i
    done

for i in $(cat accessions.txt)
    do
        fasterq-dump --threads 8 --split-files -p $i 
    done

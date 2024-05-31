#!/bin/bash

# wget https://raw.githubusercontent.com/CAST-genomics/haptools/main/example-files/1000genomes_sampleinfo.tsv
ONE_K="/dcs04/lieber/statsgen/jbenjami/resources/databases/1KG/data_raw"

cut -f 1,2 ${ONE_K}/integrated_call_samples_v3.20130502.ALL.panel | \
    sed '1d' | sed -e 's/ /\t/g' > 1k_sampleinfo.tsv

# AFR admixed
echo -e "500\tAFR_admixed\tCEU\tYRI" >> AFR_admixed.dat
echo -e "10\t0\t0.2\t0.8" >> AFR_admixed.dat

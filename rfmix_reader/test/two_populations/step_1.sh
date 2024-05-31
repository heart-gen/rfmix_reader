#!/bin/bash
#SBATCH --job-name=simu_genotypes
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jbenja13@jh.edu
#SBATCH --partition=shared,bluejay
#SBATCH --nodes=1
#SBATCH --array=21,22
#SBATCH --cpus-per-task=1
#SBATCH --mem=30gb
#SBATCH --output=output.%A_%a.log
#SBATCH --time=18:00:00

echo "**** Job starts ****"
date

echo "**** JHPCE info ****"
echo "User: ${USER}"
echo "Job id: ${SLURM_JOBID}"
echo "Job name: ${SLURM_JOB_NAME}"
echo "Node name: ${SLURM_NODENAME}"
echo "Hostname: ${HOSTNAME}"
echo "Task id: ${SLURM_ARRAY_TASK_ID}"

## List current modules for reproducibility

module load htslib
module list

## Edit with your job command
ONE_K="/dcs05/lieber/hanlab/jbenjami/resources/databases/1KG"
CHROM=${SLURM_ARRAY_TASK_ID}

echo "**** Run simulation ****"
haptools simgenotype \
	 --model AFR_admixed.dat \
	 --mapdir ${ONE_K}/genetic_maps/_m/ \
	 --chroms ${CHROM} \
	 --seed 20240126 \
	 --ref_vcf ${ONE_K}/GRCh38_phased_vcf/raw/1kGP_high_coverage_Illumina.chr${CHROM}.filtered.SNV_INDEL_SV_phased_panel.vcf.gz \
	 --sample_info 1k_sampleinfo.tsv \
	 --out ./chr${CHROM}.vcf.gz

tabix -f chr${CHROM}.vcf.gz

echo "**** Job ends ****"
date

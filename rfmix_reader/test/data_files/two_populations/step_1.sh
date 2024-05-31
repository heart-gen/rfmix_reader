#!/bin/bash
#SBATCH --job-name=simu_genotypes
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jbenja13@jh.edu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20gb
#SBATCH --output=summary.log

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

echo "**** Generate directories ****"
mkdir afr_washington
mkdir puertorico
mkdir 5k_admixed
mkdir 100_admixed

echo "**** Generate admixture models ****"
bash ../_h/step_0.sh

echo "**** Run simulation ****"
haptools simgenotype \
	 --model AFR_admixed.dat \
	 --mapdir ../../inputs/genetic_maps/_m/ \
	 --chroms 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22 \
	 --seed 20240126 \
	 --ref_vcf ../../inputs/vcf_ref/_m/1kGP_high_coverage_Illumina.chr21.filtered.SNV_INDEL_SV_phased_panel.vcf.gz \
	 --sample_info 1k_sampleinfo.tsv \
	 --out ./simulated_admixed.vcf.gz

tabix -f simulated_admixed.vcf.gz

echo "**** Job ends ****"
date

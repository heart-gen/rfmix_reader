#!/bin/bash
#SBATCH --job-name=rfmix_simu
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jbenja13@jh.edu
#SBATCH --partition=shared,bluejay
#SBATCH --nodes=1
#SBATCH --array=21,22
#SBATCH --cpus-per-task=1
#SBATCH --mem=50gb
#SBATCH --output=rfmix.%A_%a.log
#SBATCH --time=48:00:00

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

module list

## Edit with your job command
REF="/dcs05/lieber/hanlab/jbenjami/resources/databases/1KG/GRCh38_phased_vcf/refmix_ref/out"
CHROM=${SLURM_ARRAY_TASK_ID}

echo "**** Run RFMix ****"
echo -e "Chromosome: ${CRHOM}"

rfmix \
    -f chr${CHROM}.vcf.gz \
    -r $REF/1kGP_high_coverage_Illumina.chr${CHROM}.filtered.SNV_INDEL_SV_phased_panel.snpsOnly.eur.afr.vcf.gz \
    -m $REF/samples_id2 \
    -g $REF/genetic_map38 \
    -o chr${CHROM} \
    --chromosome=chr${CHROM}

echo "**** Job ends ****"
date

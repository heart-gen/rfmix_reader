#!/bin/bash
#SBATCH --job-name=simu_genotypes
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jbenja13@jh.edu
#SBATCH --partition=shared,bluejay
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5gb
#SBATCH --output=output.log

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
ONE_K="/dcs05/lieber/hanlab/jbenjami/resources/databases/1KG/data_raw"

echo "**** Generate admixture models ****"
cut -f 1,2 ${ONE_K}/integrated_call_samples_v3.20130502.ALL.panel | \
    sed '1d' | sed -e 's/ /\t/g' > 1k_sampleinfo.tsv

# AFR admixed
echo -e "500\tAFR_admixed\tCEU\tYRI" > AFR_admixed.dat
echo -e "10\t0\t0.2\t0.8" >> AFR_admixed.dat

echo "**** Job ends ****"
date

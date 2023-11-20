#$ -S /bin/bash
#$ -l tmem=4G
#$ -l h_vmem=4G
#$ -l h_rt=10:00:00
#$ -cwd
#$ -j y
#$ -N "memory_break_test"

hostname 
date
module load python/3.9.5
python3 gLV_memory_break.py
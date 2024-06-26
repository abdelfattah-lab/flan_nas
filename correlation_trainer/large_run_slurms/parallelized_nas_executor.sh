#!/bin/bash

# Create the slurm_desc directory if it doesn't exist
mkdir -p slurm_desc

log_file="job_command_mapping.log"
# Read each line from jobs.log
while IFS= read -r line; do
    # Skip lines that begin with #
    [[ $line = \#* ]] && continue
    # Extract the values from the line
    execute=$(echo "$line" | awk -F',' '{print $1}')
    job_identifier=$(echo "$line" | awk -F',' '{print $2}')
    gpu=$(echo "$line" | awk -F',' '{print $3}')
    mem=$(echo "$line" | awk -F',' '{print $4}')
    n=$(echo "$line" | awk -F',' '{print $5}')
    command=$(echo "$line" | cut -d',' -f6-)

    # Check if the execute value is True
    if [ "$execute" == "True" ]; then
        # Create a new slurm file for this job inside the slurm_desc directory
        slurm_file="slurm_desc/runJob_${job_identifier}.slurm"
        cat > "$slurm_file" <<EOL
#!/bin/bash
#SBATCH -J $job_identifier
#SBATCH -o /home/ya255/projects/iclr_nas_embedding/correlation_trainer/large_scale_run_logs/%j.out
#SBATCH -e /home/ya255/projects/iclr_nas_embedding/correlation_trainer/large_scale_run_logs/%j.err
#SBATCH -N 1
#SBATCH --mem=$mem
#SBATCH -t 32:00:00
#SBATCH --account=abdelfattah
#SBATCH --partition=abdelfattah
#SBATCH --nodelist=abdelfattah-compute-02
EOL

        # Conditionally add the GPU line
        if [ "$gpu" == "True" ]; then
            echo "#SBATCH --gres=gpu:1" >> "$slurm_file"
        fi

        # Add the n value
        echo "#SBATCH -n $n" >> "$slurm_file"

        cat >> "$slurm_file" <<EOL
export PROJ_BPATH="/home/ya255/projects/iclr_nas_embedding"

source /share/apps/anaconda3/2021.05/etc/profile.d/conda.sh 

conda activate unr

cd /home/ya255/projects/iclr_nas_embedding/nas_search

$command

echo "\${SLURM_JOB_ID},$command" >> job_command_mapping.log
EOL
    
        # Submit the slurm job
        sbatch --requeue "$slurm_file"
    fi

done < unified_nas_joblist.log
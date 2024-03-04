#!/bin/bash

# Check if a script file is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <script_file>"
    exit 1
fi

script_file=$1

# Function to submit commands from the script file
submit_commands() {
    local job_ids=()

    # Read each line in the script file
    while IFS= read -r line; do
        # Skip empty lines and comments
        [[ -z "$line" || $line == \#* ]] && continue

        # Create a temporary SLURM script
        temp_script=$(mktemp /tmp/slurm_script.XXXXXX)

        # Write the directives and setup commands into the temporary script
        cat << EOF > "$temp_script"
#!/bin/bash
#SBATCH -J allnas_nas_DARTS_fix-w-d
#SBATCH -o /home/ya255/projects/iclr_nas_embedding/CATE/large_scale_run_logs/%j.out
#SBATCH -e /home/ya255/projects/iclr_nas_embedding/CATE/large_scale_run_logs/%j.err
#SBATCH -N 1
#SBATCH --mem=50000
#SBATCH -t 16:00:00
#SBATCH --account=abdelfattah
#SBATCH --partition=abdelfattah
#SBATCH --nodelist=abdelfattah-compute-01

#SBATCH --gres=gpu:1
#SBATCH -n 2

export PROJ_BPATH="/home/ya255/projects/iclr_nas_embedding"

source /share/apps/anaconda3/2021.05/etc/profile.d/conda.sh 

conda activate cate

cd /home/ya255/projects/iclr_nas_embedding/CATE

$line
EOF

        # Submit the script to SLURM and store job ID
        job_id=$(sbatch "$temp_script" | cut -d ' ' -f 4)
        job_ids+=($job_id)

        # Remove the temporary script
        rm "$temp_script"
    done < "$script_file"

    # Wait for all jobs to complete
    for job_id in "${job_ids[@]}"; do
        srun --jobid=$job_id --wait=0
    done
}

submit_commands

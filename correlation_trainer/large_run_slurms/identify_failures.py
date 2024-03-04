import os

# Path to the completion logs and job_command_mapping.log
completion_logs_dir = os.environ['PROJ_BPATH'] + "/" + 'correlation_trainer/large_run_slurms/completion_logs'
job_command_mapping_file = "./../job_command_mapping.log"

# Read completed job IDs
completed_job_ids = set()
for filename in os.listdir(completion_logs_dir):
    if filename.endswith('_success.log'):
        job_id = filename.split('_')[0]
        completed_job_ids.add(job_id)

# Read job-command mapping
job_command_mapping = {}
with open(job_command_mapping_file, 'r') as file:
    for line in file:
        job_identifier, command = line.strip().split(',', 1)
        job_command_mapping[job_identifier] = command

# Identify and print incomplete job commands
print("List of Commands that Did Not Complete Successfully:")
for job_id, command in job_command_mapping.items():
    if job_id not in completed_job_ids:
        print(f"Job ID: {job_id}, Command: {command}")
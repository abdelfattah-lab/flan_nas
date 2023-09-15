import pandas as pd
import os

# Directories
dir_transfer = "transfer_correlation_results"
dir_base = "transfer_correlation_results_base"

# Load all CSV files from a directory into a list of dataframes
def load_csvs_from_directory(directory):
    dfs = []
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            dfs.append(pd.read_csv(os.path.join(directory, filename)))
    return dfs

# Load CSVs
transfer_dfs = load_csvs_from_directory(dir_transfer)
base_dfs = load_csvs_from_directory(dir_base)

# Process the dataframes
final_dfs = []

for transfer_df in transfer_dfs:
    for base_df in base_dfs:
        # Merge dataframes based on the matching criteria
        merged_df = pd.merge(transfer_df, base_df, left_on=['transfer_space', 'key', 'representation'], 
                             right_on=['space', 'key', 'representation'], suffixes=('', '_base'))

        # Subtract the 'kdt' values
        merged_df['kdt_diff'] = merged_df['kdt'] - merged_df['kdt_base']

        # Keep only the necessary columns
        merged_df = merged_df[['transfer_space', 'key', 'representation', 'kdt_diff']]
        final_dfs.append(merged_df)

# Concatenate all processed dataframes vertically
final_df = pd.concat(final_dfs, axis=0, ignore_index=True)

# Convert the final dataframe to LaTeX format
latex_code = final_df.to_latex(index=False, escape=False)

print(latex_code)


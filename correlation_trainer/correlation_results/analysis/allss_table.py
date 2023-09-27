import os
import pandas as pd
import re
rep_mapping = {
        'arch2vec': 'UGAE',
        'cate': 'UTAE',
        'zcp': 'ZCP',
        'adj_mlp': 'MLP',
        'adj_gin': 'UGN',
        'adj_gin_cate': 'UGN$_{UTAE}$',
        'adj_gin_arch2vec': 'UGN$_{UGAE}$',
        'adj_gin_zcp': 'UGN$_{ZCP}$',  # No change specified for this one
        'adj_gin_a2vcatezcp': 'UGN$_{CAZ}$'  # No change specified for this one
    }

# Function to surround value with dollar signs
def surround_with_dollar(value):
    # Updated regex to match negative numbers as well
    if re.match(r"(-?\d+\.\d+_\{-?\d+\.\d+\})", str(value)) or isinstance(value, (int, float)):
        return f"${value}$"
    return value


# Initialize an empty DataFrame to hold the final results
final_df = pd.DataFrame()

# Loop through each folder in the current directory
for folder in os.listdir():
    if 'allss' in folder:
        # Loop through each CSV file in the folder
        for file in os.listdir(folder):
            if file.endswith('.csv'):
                file_path = os.path.join(folder, file)
                df = pd.read_csv(file_path)
                
                # Determine the space column
                space_col = 'transfer_space' if folder.endswith('t') else 'space'
                
                for index, row in df.iterrows():
                    representation = rep_mapping.get(row['representation'], row['representation'])
                    if folder.endswith('t'):
                        representation = representation.replace("$_{", "$^{T}$$_{")
                    
                    # Replace 'UGN' with 'FLAN' in representation
                    representation = representation.replace('UGN', 'FLAN')
                    
                    key = row['key']
                    space = row[space_col]
                    key = row['key']
                    space = row[space_col]
                    space = space.replace('_fix-w-d', '$_{FixWD}$').replace('_lr-wd', '$_{LRWD}$')

                    kdt_with_std = f"{row['kdt']:.4f}_{{{row['kdt_std']:.4f}}}"
                    
                    final_df = final_df.append({
                        'Samples': key,
                        'Space': space,
                        'Representation': representation,
                        'KDT': surround_with_dollar(kdt_with_std)
                    }, ignore_index=True)

# Process to add '\bM' prefix to the highest KDT within each 'Samples' column for each 'Space'
final_df['kdt'] = final_df.apply(lambda row: float(re.search(r"(-?\d+\.\d+)", row['KDT']).group()), axis=1)

def add_prefix(row):
    max_idx = row['kdt'].idxmax()
    row['KDT'][max_idx] = r'\bM' + row['KDT'][max_idx]
    return row

final_df = final_df.groupby(['Space', 'Samples']).apply(add_prefix)

# Drop the temporary 'kdt' column used for comparison
final_df.drop(columns=['kdt'], inplace=True)

# Pivot the DataFrame to get the desired format
final_df = final_df.pivot_table(index=['Space', 'Representation'], columns='Samples', values='KDT', aggfunc='first')

# Convert the DataFrame to LaTeX format and print
print(final_df.to_latex(escape=False))
with open('allss.tex', 'w') as f:
    f.write(final_df.to_latex(escape=False))
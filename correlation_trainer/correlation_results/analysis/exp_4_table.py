import pandas as pd
import os, re

# Directory
dir_table4 = "correlation_results/table4"

def surround_with_dollar(value):
    # Check if value matches the pattern 0.2002_{0.0001} or is a number
    if re.match(r"(\d+\.\d+_\{\d+\.\d+\})", str(value)) or isinstance(value, (int, float)):
        return f"${value}$"
    return value

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

mode_mapping = {
    'dense': '$^{dgf}_{dgf}$',
    'ensemble': '$^{dgf.gat}_{dgf.gat}$',
    'gat': '$^{gat}_{gat}$',
    'gat_mh': '$%{mhgat}_{mhgat}$',
}


# Load all CSV files from a directory into a list of dataframes
def load_csvs_from_directory(directory):
    dfs = []
    for filename in os.listdir(directory):
        if filename.endswith("_samp_eff.csv"):
            dfs.append(pd.read_csv(os.path.join(directory, filename)))
    return dfs

# Load CSVs
exp1_dfs = load_csvs_from_directory(dir_table4)

# Concatenate all dataframes
df = pd.concat(exp1_dfs, axis=0, ignore_index=True)

# Replace representation names
df['representation'] = df['representation'].map(rep_mapping)

# Create a new column combining 'representation', 'gnn_type', and 'back_dense'
def create_column_name(row):
    if row['representation'] in ["arch2vec", "cate", "zcp", "adj_mlp"]:
        return row['representation']
    else:
        name = f"{row['representation']}{mode_mapping[row['gnn_type']]}"
        if row['back_dense']:
            # name += "_back_dense"
            name = name[:-6] + "}$"
        return name

df['Samples'] = df.apply(create_column_name, axis=1)

# Pivot the table to get the desired structure
df['kdt_with_std'] = df.apply(lambda row: f"{row['kdt']:.4f}_{{{row['kdt_std']:.4f}}}", axis=1)
# final_df = df.pivot_table(index=['space', 'key'], columns='combined_col', values=['kdt', 'kdt_std'], aggfunc='first')

final_df = df.pivot_table(index=['transfer_space', 'key'], columns='Samples', values='kdt_with_std', aggfunc='first')

# Format numbers to 4 decimal places
final_df = final_df.applymap(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)

final_df = final_df.applymap(surround_with_dollar)

final_df = final_df.replace(to_replace=r"_fix-w-d", value="$_{FixWD}$", regex=True)
final_df = final_df.replace(to_replace=r"_lr-w-d", value="$_{LRWD}$", regex=True)



# # List of column names in the desired order
# desired_order = ['UGN', 'UGN$_{UTAE}$', 'UGN$_{UGAE}$', 'UGN$_{ZCP}$', 'UGAE', 'UTAE', 'ZCP', 'MLP']

# # Ensure that all columns in desired_order exist in the DataFrame
# final_columns = [col for col in desired_order if col in final_df.columns]

# # Reorder the DataFrame columns
# final_df = final_df[final_columns]


# Convert the final dataframe to LaTeX format
latex_code = final_df.to_latex(index=True, escape=False, multirow=True, multicolumn=True, multicolumn_format='c')

print(latex_code)
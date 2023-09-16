import pandas as pd
import os, re

# Directory
dir_exp2 = "correlation_results/exp2"

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
        'adj_gin_zcp': 'UGN$_{ZCP}$'  # No change specified for this one
    }

mode_mapping = {
    'dense': '$_{DGF}$',
    'ensemble': '$_{DGF.GAT}$',
    'gat': '$_{GAT}$',
    'gat_mh': '$_{MHGAT}$',
}
# Create a new column combining 'representation', 'gnn_type', and 'back_dense'
def create_column_name(row):
    if row['representation'] in ["arch2vec", "cate", "zcp", "adj_mlp"]:
        return row['representation']
    else:
        name = f"{row['representation']}{mode_mapping[row['gnn_type']]}"
        if row['back_dense']:
            # name += "_back_dense"
            name = name[:-2] + "-bd}$"
        return name


# Load all CSV files from a directory into a list of dataframes
def load_csvs_from_directory(directory):
    dfs = []
    for filename in os.listdir(directory):
        if filename.endswith("_samp_eff.csv"):
            dfs.append(pd.read_csv(os.path.join(directory, filename)))
    return dfs

# Load CSVs
exp2_dfs = load_csvs_from_directory(dir_exp2)

# Concatenate all dataframes
df = pd.concat(exp2_dfs, axis=0, ignore_index=True)

# Replace representation names
df['representation'] = df['representation'].map(rep_mapping)

# Create a new column combining 'representation', 'gnn_type', and 'back_dense'
df['Samples'] = df.apply(create_column_name, axis=1)

# Create a new column for the varying quantity
def determine_varying_quantity(row):
    if pd.notna(row['timesteps']):
        return f"timesteps_{row['timesteps']}"
    else:
        return create_column_name(row)

df['Varying_Quantity'] = df.apply(determine_varying_quantity, axis=1)

# Filter data for timesteps study
df_timesteps = df[pd.notna(df['timesteps'])]

# Pivot the table for timesteps study
df_timesteps['kdt_with_std'] = df_timesteps.apply(lambda row: f"{row['kdt']:.4f}_{{{row['kdt_std']:.4f}}}", axis=1)
final_df_timesteps = df_timesteps.pivot_table(index=['space', 'Varying_Quantity'], columns='Samples', values='kdt_with_std', aggfunc='first')
final_df_timesteps = final_df_timesteps.replace(to_replace=r"_fix-w-d", value="$_{FixWD}$", regex=True)
final_df_timesteps = final_df_timesteps.replace(to_replace=r"_lr-w-d", value="$_{LRWD}$", regex=True)
final_df_timesteps = final_df_timesteps.applymap(surround_with_dollar)
# for final_df_timesteps, only keep the 'UGN$_{DGF}$' column
final_df_timesteps = final_df_timesteps[['UGN$_{DGF}$']]

# Reset the index
final_df_timesteps = final_df_timesteps.reset_index()

# Pivot the table to have timesteps as rows and search space as columns
final_df_timesteps = final_df_timesteps.pivot(index='Varying_Quantity', columns='space', values='UGN$_{DGF}$')


# Filter data for gnn_type and back_dense study where timesteps is 2
df_gnn = df[(df['timesteps'] == 2) & (pd.isna(df['timesteps']) == False)]

# Pivot the table for gnn_type and back_dense study
df_gnn['kdt_with_std'] = df_gnn.apply(lambda row: f"{row['kdt']:.4f}_{{{row['kdt_std']:.4f}}}", axis=1)
final_df_gnn = df_gnn.pivot_table(index=['space', 'Varying_Quantity'], columns='Samples', values='kdt_with_std', aggfunc='first')
final_df_gnn = final_df_gnn.applymap(surround_with_dollar)
final_df_gnn = final_df_gnn.replace(to_replace=r"_fix-w-d", value="$_{FixWD}$", regex=True)
final_df_gnn = final_df_gnn.replace(to_replace=r"_lr-w-d", value="$_{LRWD}$", regex=True)

# Convert the dataframes to LaTeX format
latex_code_timesteps = final_df_timesteps.to_latex(index=True, escape=False, multirow=True, multicolumn=True, multicolumn_format='c')
latex_code_gnn = final_df_gnn.to_latex(index=True, escape=False, multirow=True, multicolumn=True, multicolumn_format='c')

print("Timesteps Study Table:\n", latex_code_timesteps)
print("\nGNN Type and Back Dense Study Table:\n", latex_code_gnn)
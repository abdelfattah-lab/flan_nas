import os
import pandas as pd

folder_path = './'

# Iterate over all files in the folder
fr_cache = []
for file_name in os.listdir(folder_path):
    maxval = {}
    if file_name.endswith('_samp_eff.csv'):
        file_path = os.path.join(folder_path, file_name)

        # Read the file using pandas
        df = pd.read_csv(file_path)

        # Start a new table with space name
        space_name = file_name.split('_samp_eff.csv')[0]
        latex_table = "\\begin{tabular}{lcccccc"

        # Add columns for each unique key
        unique_keys = df['key'].unique()
        for key in unique_keys:
            latex_table += "c"
        latex_table += "}\n"

        header_row = "timesteps & Res & LeakR & UnqAtn & OpAtn & AtnRsc"
        for key in unique_keys:
            header_row += f" & kdt ({key})"
        header_row += "\\\\ \\hline\n"

        latex_table += "\\multicolumn{" + str(len(unique_keys) + 6) + "}{c}{" + space_name + "}\\\\ \\hline\n"
        latex_table += header_row

        for timesteps in df['timesteps'].unique():
            df_filtered = df[df['timesteps'] == timesteps]
            
            # Iterate through each row in filtered DataFrame
            for index, row in df_filtered.iterrows():
                latex_row = f"{row['timesteps']} & "
                latex_row += '\\ding{51}' if row['residual'] else '\\ding{55}'
                latex_row += ' & ' + ('\\ding{51}' if row['leakyrelu'] else '\\ding{55}')
                latex_row += ' & ' + ('\\ding{51}' if row['unique_attention_projection'] else '\\ding{55}')
                latex_row += ' & ' + ('\\ding{51}' if row['opattention'] else '\\ding{55}')
                latex_row += ' & ' + ('\\ding{51}' if row['attention_rescale'] else '\\ding{55}')
                # find all rows of df_filtered that satisfy the conditions above
                frows = df_filtered[(df_filtered['residual'] == row['residual']) & (df_filtered['leakyrelu'] == row['leakyrelu']) & (df_filtered['unique_attention_projection'] == row['unique_attention_projection']) & (df_filtered['opattention'] == row['opattention']) & (df_filtered['attention_rescale'] == row['attention_rescale'])]
                for k_ in frows.key:
                    kdtfrow = frows[frows['key'] == k_]['kdt'].item()
                    maxval[k_] = max(maxval.get(k_, 0), kdtfrow)
                    latex_row += f" & {kdtfrow:.4f}"
                latex_row += "\\\\\n"
                if kdtfrow in fr_cache:
                    pass
                else:
                    fr_cache.append(kdtfrow)
                    latex_table += latex_row

        latex_table += "\\end{tabular}"
        # In table, find the max value for each key and highlight it
        for key in unique_keys:
            latex_table = latex_table.replace(f"{maxval[key]:.4f}", f"\\textbf{{{maxval[key]:.4f}}}")
        # Save the LaTeX table to a file, each space in its own file.
        with open(f'latex_table_{space_name}.tex', 'w') as file:
            file.write(latex_table)

        print(latex_table + '\n\n')

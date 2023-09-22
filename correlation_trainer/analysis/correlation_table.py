import pandas as pd
import re


def surround_with_dollar(value):
    # Check if value matches the pattern 0.2002_{0.0001} or is a number
    if re.match(r"(\d+\.\d+_\{\d+\.\d+\})", str(value)) or isinstance(value, (int, float)):
        return f"${value}$"
    return value


def generate_latex_table(spaces_subset):
    # Initialize an empty dataframe for the final table
    final_df = pd.DataFrame()
    final_df_list = []
    rep_mapping = {
        'adj_gin': 'UGN',
        'adj_gin_cate': 'UGN$_{UTAE}$',
        'adj_gin_arch2vec': 'UGN$_{UGAE}$',
        'adj_gin_zcp': 'UGN$_{ZCP}$'  # No change specified for this one
    }

    # Loop through each space in the subset
    for space in spaces_subset:
        if space.__class__.__name__=='tuple':
            df = pd.read_csv(f"correlation_results/sample_efficiency_results/{space[0]}_samp_eff.csv")
            df = df[df['task'] == space[1]]
            space = space[0]
        else:
            # Read the CSV file for the current space
            df = pd.read_csv(f"correlation_results/sample_efficiency_results/{space}_samp_eff.csv")

        # Filter the dataframe for the current space
        df_space = df[df['space'] == space]
        # Create a new dataframe for the current space
        space_df = df_space[['key']].drop_duplicates().reset_index(drop=True)

        # Loop through each representation
        for old_rep, new_rep in rep_mapping.items():
            # Filter dataframe for the current representation
            df_rep = df_space[df_space['representation'] == old_rep]
            
            
            # Create a new column combining 'kdt' and 'kdt_std' in the desired format with 4 decimal precision
            df_rep = df_rep[['key', 'kdt', 'kdt_std']].drop_duplicates().reset_index(drop=True)
            df_rep[new_rep] = df_rep.apply(lambda row: f"{row['kdt']:.4f}_{{{row['kdt_std']:.4f}}}", axis=1)
            
            # Merge the representation data into the space dataframe
            space_df = pd.merge(space_df, df_rep[['key', new_rep]], on='key', how='outer')
        print(space_df)
        # Drop the 'key' column for all but the first space to avoid repetition
        # if final_df_list:
        #     space_df = space_df.drop(columns='key')

        # Append the space dataframe to the final dataframe list
        final_df_list.append(space_df)

    # Concatenate all space dataframes vertically
    # final_df = pd.concat(final_df_list, axis=0, ignore_index=True)

    # Concatenate all space dataframes vertically
    concatenated_dfs = []
    for space_df, space in zip(final_df_list, spaces_subset):
        if space.__class__.__name__=='tuple':
            space = space[0] + ' ' + space[1]
        # Create a dataframe for the space name
        # space_name_df = pd.DataFrame({space: ["Search Space: " + space]}, columns=space_df.columns)
        space_name_df = pd.DataFrame(columns=space_df.columns)
        # Fill in the first column (assuming it's the name of the space) with the search space name
        space_name_df.at[0, space_df.columns[0]] = "Search Space: " + space
        # fill rest with 0
        space_name_df = space_name_df.fillna(0)
        # import pdb; pdb.set_trace()
        # print(space)
        
        # Concatenate the space name dataframe with the space dataframe
        concatenated_dfs.append(pd.concat([space_name_df, space_df], axis=0, ignore_index=True))
    # import pdb; pdb.set_trace()
    final_df = pd.concat(concatenated_dfs, axis=0, ignore_index=True)
    final_df = final_df.applymap(surround_with_dollar)

    final_df = final_df.replace(to_replace=r"_fix-w-d", value="$_{FixWD}$", regex=True)
    final_df = final_df.replace(to_replace=r"_lr-w-d", value="$_{LRWD}$", regex=True)

    # Convert the final dataframe to LaTeX format
    latex_code = final_df.to_latex(index=False, escape=False, na_rep="")
    # latex_code = latex_code.replace('_', '$_{').replace('}', '}$')
    return latex_code


# Specify the subset of spaces you want to include in the table
spaces_subset = ['nb101', 'nb201', 'DARTS', 'DARTS_fix-w-d', 'ENAS', 'ENAS_fix-w-d', ('tb101', 'segmentsemantic')]

# Generate the LaTeX table code
latex_table = generate_latex_table(spaces_subset)
print(latex_table)

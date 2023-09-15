import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create the directory if it doesn't exist
if not os.path.exists('correlation_results/transfer_correlation_graphs'):
    os.makedirs('correlation_results/transfer_correlation_graphs')

# Directories
dir_transfer = "correlation_results/transfer_correlation_results"
dir_base = "correlation_results/transfer_correlation_results_base"

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
for transfer_df in transfer_dfs:
    unique_spaces = transfer_df['transfer_space'].unique()
    for space in unique_spaces:
        # Filter the dataframe for the current space
        df_space = transfer_df[transfer_df['transfer_space'] == space]
        # Find the corresponding base dataframe
        base_df_space = None
        for base_df in base_dfs:
            if space in base_df['space'].unique():
                base_df_space = base_df
                break

        # Set the style of seaborn
        sns.set_style("whitegrid")

        # Create a figure and axis
        plt.figure(figsize=(10, 6))

        # Get unique representations for the current space
        representations = df_space['representation'].unique()

        # Loop through each representation
        for rep in representations:
            # Filter dataframe for the current representation
            df_rep = df_space[df_space['representation'] == rep]

            # Plot with shaded variance for transfer data
            plt.fill_between(df_rep['key'], df_rep['kdt'] - df_rep['kdt_std'], df_rep['kdt'] + df_rep['kdt_std'], alpha=0.2)
            plt.plot(df_rep['key'], df_rep['kdt'], label=f"{rep} (Transfer)")

            # If a corresponding base dataframe exists, overlay its plot
            if base_df_space is not None:
                base_df_rep = base_df_space[base_df_space['representation'] == rep]
                base_df_rep = base_df_rep.sort_values(by='key')
                plt.plot(base_df_rep['key'], base_df_rep['kdt'], linestyle='--', label=f"{rep} (Base)")


        # Set labels and title
        plt.xlabel('Number Of Training Samples')
        plt.ylabel('Kendall Tau Correlation')
        plt.title(f'Kendall Tau Correlation for {space}')
        plt.legend()

        # Save the plot in both PNG and PDF formats
        plt.savefig(f"correlation_results/transfer_correlation_graphs/{space}.png")
        plt.savefig(f"correlation_results/transfer_correlation_graphs/{space}.pdf")

        # Close the plot to free up memory
        plt.close()

print("Graphs saved in 'correlation_graphs' folder.")



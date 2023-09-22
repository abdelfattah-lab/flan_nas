import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

base_path = 'correlation_results/correlation_graphs'

# Create the directory if it doesn't exist
if not os.path.exists('correlation_results/correlation_graphs'):
    os.makedirs('correlation_results/correlation_graphs')

# List of spaces
spaces = ['Amoeba', 'DARTS_fix-w-d', 'DARTS_lr-wd', 'DARTS', 'ENAS_fix-w-d', 'ENAS', 'NASNet', 'PNAS_fix-w-d', 'PNAS', 'nb101', 'nb201', 'nb301', 'tb101']

def plot_graph(df, space, task=None):
    # Set the style of seaborn
    sns.set_style("whitegrid")

    # Create a figure and axis
    plt.figure(figsize=(10, 6))

    # Loop through each representation
    for rep in ['adj_gin', 'adj_gin_cate', 'adj_gin_arch2vec', 'adj_gin_zcp']:
        # Filter dataframe for the current representation
        df_rep = df[df['representation'] == rep]

        # Sort dataframe by 'key'
        df_rep = df_rep.sort_values(by='key')

        # Plot with shaded variance
        plt.fill_between(df_rep['key'], df_rep['kdt'] - df_rep['kdt_std'], df_rep['kdt'] + df_rep['kdt_std'], alpha=0.2)
        plt.plot(df_rep['key'], df_rep['kdt'], label=rep)

    # Set labels and title
    plt.xlabel('Number Of Training Samples')
    plt.ylabel('Kendall Tau Correlation')
    # Set log scale
    plt.xscale('log', basex=2)
    # Set legend
    plt.legend()
    if task:
        plt.title(f'Kendall Tau Correlation for {space} - Task: {task}')
        # Save the plot in both PNG and PDF formats
        plt.savefig(f"correlation_results/correlation_graphs/{space}_{task}.png")
        plt.savefig(f"correlation_results/correlation_graphs/{space}_{task}.pdf")
    else:
        plt.title(f'Kendall Tau Correlation for {space}')
        # Save the plot in both PNG and PDF formats
        plt.savefig(f"correlation_results/correlation_graphs/{space}.png")
        plt.savefig(f"correlation_results/correlation_graphs/{space}.pdf")

# Loop through each space
for space in spaces:
    # Read the CSV file for the current space
    # df = pd.read_csv(f"{space}.csv")
    print(f"Reading CSV for {space}")
    try:
        df = pd.read_csv(f"correlation_results/sample_efficiency_results/{space}_samp_eff.csv")
    except Exception as e:
        print(f"Error for {space}: {e}")
        continue

    # Filter the dataframe for the current space
    df_space = df[df['space'] == space]

    # If the space is 'tb101', loop through each task
    if space == 'tb101':
        try:
            print("Plotting graph for ", space)
            tasks = df_space['task'].unique()
            for task in tasks:
                df_task = df_space[df_space['task'] == task]
                plot_graph(df_task, space, task=task)
            print("Plotted graph for ", space)
        except Exception as e:
            print(f"Error for {space}: {e}")
    else:
        try:
            print("Plotting graph for ", space)
            plot_graph(df_space, space)
            print("Plotted graph for ", space)
        except Exception as e:
            print(f"Error for {space}: {e}")

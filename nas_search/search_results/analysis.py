import pandas as pd
import matplotlib.pyplot as plt
import os

# Define the experiment folders and their corresponding suffixes
experiments = {
    "exp7": "_t",
    "exp6": "_s",
    "exp8": "_ut"
}

# Create the search_graphs directory if it doesn't exist
if not os.path.exists("search_graphs"):
    os.makedirs("search_graphs")

# Get the list of csv files from the first experiment folder as a reference
reference_csv_files = [f for f in os.listdir(next(iter(experiments))) if f.endswith("_search_eff.csv")]

# Loop through each csv file (each space)
for csv_file in reference_csv_files:
    space = csv_file.replace("_search_eff.csv", "")
    
    # Create a new figure for the space
    plt.figure()

    # Loop through each experiment folder
    for exp, suffix in experiments.items():
        file_path = os.path.join(exp, csv_file)
        
        # Check if the file exists
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)

            # Plot for each representation
            representations = df['representation'].unique()
            for representation in representations:
                subset = df[df['representation'] == representation]
                plt.plot(subset['num_samps'], subset['av_best_acc'], label=f"{representation}{suffix}")
                
                # Add shaded error regions
                plt.fill_between(subset['num_samps'], 
                                subset['av_best_acc'] - subset['best_acc_std'], 
                                subset['av_best_acc'] + subset['best_acc_std'], 
                                alpha=0.2)

    # Beautify the plot
    plt.title(f"Results for {space}")
    plt.xlabel("Number of Samples")
    plt.xlim([4, 64])
    plt.ylim(bottom=0.9)
    # plt.yscale("log")
    plt.xscale("log", basex=2)
    plt.ylabel("Average Best Accuracy")
    plt.legend()
    plt.tight_layout()

    # Save the plot
    plt.savefig(f"search_graphs/{space}.png")
    plt.savefig(f"search_graphs/{space}.pdf")
    plt.close()

print("Graphs saved in 'search_graphs' folder.")

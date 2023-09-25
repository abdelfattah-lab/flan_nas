import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import matplotlib.ticker as ticker

# Set a consistent color palette
sns.set_palette("tab10")

# Enable LaTeX interpretation in matplotlib
plt.rcParams['text.usetex'] = False
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = 12  # Increase font size

space_map = {'nb101': "NASBench-101", "nb201": "NASBench-201", "ENAS_fix-w-d": "ENAS$_{FixWD}$"}

# Define the experiment folders and their corresponding suffixes
experiments = {
    "exp7": "$^{T}$",
    "exp6": "",
    "exp8": "$^{UT}$"
}
exp6_representations = {"nb101": ['adj_gin', 'adj_gin_zcp'],
                        "nb201": ['adj_gin', 'adj_gin_zcp'],
                        "ENAS_fix-w-d": ['adj_gin', 'adj_gin_zcp', 'adj_gin_cate']
                        }
exp7_representations = {"nb101": ['adj_gin', 'adj_gin_zcp'],
                        "nb201": ['adj_gin', 'adj_gin_zcp'],
                        "ENAS_fix-w-d": ['adj_gin', 'adj_gin_zcp']
                        }
exp8_representations = {"nb101": ['adj_gin_zcp'],
                        "nb201": ['adj_gin_cate'],
                        "ENAS_fix-w-d": []
                        }
representation_map = {
    'adj_gin': 'UGN',
    'adj_gin_zcp': 'UGN$_{ZCP}$',
    'adj_gin_cate': 'UGN$_{CATE}$',
    'adj_gin_arch2vec': 'UGN$_{Arch2Vec}$',
}

lim_map = {
    "nb201": {"y": (0.875, 0.925), "x": (4, 192)},
    "ENAS_fix-w-d": {"y": (0.875, 1.01), "x": (4, 192)},
    "nb101": {"y": (0.8, 1.01), "x": (4, 192)},
}

# Create the search_graphs directory if it doesn't exist
if not os.path.exists("search_graphs"):
    os.makedirs("search_graphs")

# Get the list of csv files from the first experiment folder as a reference
reference_csv_files = [f for f in os.listdir(next(iter(experiments))) if f.endswith("_search_eff.csv")]


# List of spaces to analyze
spaces_to_analyze = ['nb101', 'nb201', 'ENAS_fix-w-d']  # Modify this list as needed

# Create a single figure with three subplots
fig, axes = plt.subplots(1, 3, figsize=(21, 5))


# Loop through each space
for idx, space in enumerate(spaces_to_analyze):
    ax = axes[idx]

    # Set plot limits
    ax.set_xlim(lim_map[space]['x'])
    ax.set_ylim(lim_map[space]['y'])

    # Loop through each experiment folder
    for exp, suffix in experiments.items():
        file_name = f"{space}_search_eff.csv"
        file_path = os.path.join(exp, file_name)

        # Check if the file exists
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
                # Stripping spaces from string columns
            df = df.apply(lambda col: col.str.strip() if col.dtype == 'object' else col)
            
            # Get the representations to be plotted for the current experiment and space
            if exp == "exp6":
                reps_to_plot = exp6_representations.get(space, [])
            if exp == "exp7":
                reps_to_plot = exp7_representations.get(space, [])
            if exp == "exp8":
                reps_to_plot = exp8_representations.get(space, [])
            reps_to_plot = [r for r in reps_to_plot if r in df['representation'].unique()]

            # Plot for each specified representation
            for representation in reps_to_plot:
                mapped_representation = representation_map.get(representation, representation)
                subset = df[df['representation'] == representation]
                ax.plot(subset['num_samps'], subset['av_best_acc'], label=f"{mapped_representation}{suffix}", marker='o', markersize=4)

                # Add shaded error regions
                ax.fill_between(subset['num_samps'],
                                subset['av_best_acc'] - subset['best_acc_std'],
                                [min(1, x) for x in subset['av_best_acc'] + subset['best_acc_std']],
                                alpha=0.1)


    # Beautify the plot
    ax.set_title(f"NAS Search on {space_map[space]}")
    ax.set_xlabel("Number of Samples")
    ax.set_xscale('log', base=2)
    ax.get_xaxis().set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))
    ax.get_xaxis().set_minor_formatter(ticker.StrMethodFormatter("{x:.0f}"))
    ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
    ax.set_ylabel("Average Best Accuracy")
    ax.legend(loc='lower right', fontsize=8)  # This ensures each subplot has its own legend
    ax.grid(True, which="both", ls="--", c='0.7')  # Add a grid for better readability

# Adjust the layout to make it tight
plt.tight_layout()

# Save the entire figure containing all three subplots
plt.savefig("search_graphs/combined_spaces.png")
plt.savefig("search_graphs/combined_spaces.pdf")

print("Graphs saved in 'search_graphs' folder.")

# import pandas as pd
# import matplotlib.pyplot as plt
# import os
# import seaborn as sns

# # Set a consistent color palette
# sns.set_palette("tab10")

# # Enable LaTeX interpretation in matplotlib
# plt.rcParams['text.usetex'] = False
# plt.rcParams['mathtext.fontset'] = 'cm'
# plt.rcParams['font.size'] = 6  # Increase font size


# # Define the experiment folders and their corresponding suffixes
# experiments = {
#     "exp7": "$^{T}$",
#     "exp6": "",
#     "exp8": "$^{UT}$"
# }

# representation_map = {
#     'adj_gin': 'UGN',
#     'adj_gin_zcp': 'UGN$_{ZCP}$',
#     'adj_gin_cate': 'UGN$_{CATE}$',
#     'adj_gin_arch2vec': 'UGN$_{Arch2Vec}$',
# }

# lim_map = {
#     "nb201": {"y": (0.875, 0.925), "x": (2, 256)},
#     "ENAS": {"y": (0.875, 0.925), "x": (2, 256)},
#     "nb101": {"y": (0.8, 0.925), "x": (2, 256)},
# }

# # Create the search_graphs directory if it doesn't exist
# if not os.path.exists("search_graphs"):
#     os.makedirs("search_graphs")

# # Get the list of csv files from the first experiment folder as a reference
# reference_csv_files = [f for f in os.listdir(next(iter(experiments))) if f.endswith("_search_eff.csv")]

# # Loop through each csv file (each space)
# for csv_file in reference_csv_files:
#     space = csv_file.replace("_search_eff.csv", "")
    
#     # Create a new figure for the space
#     plt.figure()

#     # Loop through each experiment folder
#     for exp, suffix in experiments.items():
#         file_path = os.path.join(exp, csv_file)
        
#         # Check if the file exists
#         if os.path.exists(file_path):
#             df = pd.read_csv(file_path)

#             # Plot for each representation
#             representations = df['representation'].unique()
#             for representation in representations:
#                 mapped_representation = representation_map[representation.replace(" ", "")]
#                 subset = df[df['representation'] == representation]
#                 plt.plot(subset['num_samps'], subset['av_best_acc'], label=f"{mapped_representation}{suffix}")
                
#                 # Add shaded error regions
#                 plt.fill_between(subset['num_samps'], 
#                                 subset['av_best_acc'] - subset['best_acc_std'], 
#                                 [min(1, x) for x in subset['av_best_acc'] + subset['best_acc_std']], 
#                                 alpha=0.1)

#     # Beautify the plot
#     plt.title(f"Results for {space}")
#     plt.xlabel("Number of Samples")
#     # plt.xlim([4, 64])
#     plt.ylim([0.8, 1])
#     # plt.yscale("log")
#     plt.xscale("log", basex=2)
#     plt.ylabel("Average Best Accuracy")
#     plt.legend()
#     plt.tight_layout()
#     plt.grid(True, which="both", ls="--", c='0.7')  # Add a grid for better readability


#     # Save the plot
#     plt.savefig(f"search_graphs/{space}.png")
#     plt.savefig(f"search_graphs/{space}.pdf")
#     plt.close()

# print("Graphs saved in 'search_graphs' folder.")

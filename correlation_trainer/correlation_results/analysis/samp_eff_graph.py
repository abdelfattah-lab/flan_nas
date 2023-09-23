
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
plt.rcParams['font.size'] = 14  # Increase font size


tagates_eff = {
    'nb101': dict(zip([72, 364, 729, 3645, 7290], [0.6686, 0.7744, 0.7839, 0.8133, 0.8217])),
    'nb201': dict(zip([39, 78, 390, 781], [0.5382, 0.6707, 0.7731, 0.8660, 0.8890])),
    'ENAS': dict(zip([25, 50, 125, 250, 500], [0.3458, 0.4407, 0.5485, 0.6324, 0.6683])),
}


gcn_eff = {
    'nb101': dict(zip([72, 364, 729, 3645, 7290], [0.3668, 0.5973, 0.6927, 0.7520, 0.7689])),
    'nb201': dict(zip([7, 39, 78, 390, 781], [0.2461, 0.3113, 0.4080, 0.5461, 0.6095])),
    'ENAS': dict(zip([25, 50, 125, 250, 500], [0.2301, 0.3140, 0.3367, 0.3508, 0.3715])),
}

multipredict_eff = {
    'nb101': dict(zip([72, 364, 729], [0.5918, 0.6626, 0.6842])),
    'nb201': dict(zip([7, 39, 78], [0.2483, 0.3971, 0.3761])),
    'ENAS': dict(zip([25, 50, 125], [0.3873, 0.4586, 0.5407])),
}




# List of spaces to analyze
spaces_to_analyze = ['nb101', 'nb201', 'ENAS']  # Add other spaces to this list as needed


# Now reading 'tagates_eff' from CSV files and updating the hardcoded tagates_eff
for space in spaces_to_analyze:
    file_path = f'tagates_base/{space}_final_corr.csv'
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        # Update the corresponding 'space' dictionary with the CSV values
        tagates_eff[space].update(dict(zip(df.iloc[:, 0], df.iloc[:, 1])))
    else:
        print(f"{file_path} does not exist, please make sure the file path is correct.")

# For each space, sort the dictionary by key
for space in spaces_to_analyze:
    if tagates_eff.get(space):
        tagates_eff[space] = dict(sorted(tagates_eff[space].items()))

# Experiment folders and their corresponding suffixes
experiments = {
    "table3": "",
    "table4": "$^{T}$"
}

space_map = {'nb101': "NASBench-101", "nb201": "NASBench-201", "ENAS": "ENAS"}



# exp3_representations = {"nb101": [('adj_gin_zcp', (0, 1024))],
#                         "nb201": [('adj_gin', (0, 1024)), ('adj_gin_arch2vec', (0, 1024))],
#                         "ENAS": [('adj_gin_zcp', (0, 1024))]
#                         }
# exp4_representations = {"nb101": [('adj_gin', (0, 64))],
#                         "nb201": [],
#                         "ENAS": [('adj_gin_zcp', (0, 1024))]
#                         }

# exp3_representations = {"nb101": ['adj_gin', 'adj_gin_zcp', 'adj_gin_cate', 'adj_gin_arch2vec'],
#                         "nb201": ['adj_gin', 'adj_gin_zcp', 'adj_gin_cate', 'adj_gin_arch2vec'],
#                         "ENAS": ['adj_gin', 'adj_gin_zcp', 'adj_gin_cate', 'adj_gin_arch2vec']
#                         }
# exp4_representations = {"nb101":  ['adj_gin', 'adj_gin_zcp', 'adj_gin_cate', 'adj_gin_arch2vec'],
#                         "nb201":  ['adj_gin', 'adj_gin_zcp', 'adj_gin_cate', 'adj_gin_arch2vec'],
#                         "ENAS": ['adj_gin', 'adj_gin_zcp', 'adj_gin_cate', 'adj_gin_arch2vec']
#                         }
exp3_representations = {"nb101": ['adj_gin', 'adj_gin_zcp', 'adj_gin_cate', 'adj_gin_arch2vec'],
                        "nb201": ['adj_gin', 'adj_gin_zcp', 'adj_gin_cate', 'adj_gin_arch2vec'],
                        "ENAS": ['adj_gin', 'adj_gin_zcp', 'adj_gin_cate', 'adj_gin_arch2vec']
                        }
exp4_representations = {"nb101":  ['adj_gin', 'adj_gin_zcp', 'adj_gin_cate', 'adj_gin_arch2vec'],
                        "nb201":  ['adj_gin', 'adj_gin_zcp', 'adj_gin_cate', 'adj_gin_arch2vec'],
                        "ENAS": ['adj_gin', 'adj_gin_zcp', 'adj_gin_cate', 'adj_gin_arch2vec']
                        }
pltlims = {
    "nb101": {'x': (1, 128), 'y': (0.1, 0.85)},
    "nb201": {'x': (1, 128), 'y': (0.1, 0.95)},
    "ENAS":  {'x': (1, 128), 'y': (0.1, 0.8)}
}

representation_map = {
    'adj_gin': 'FLAN',
    'adj_gin_zcp': 'FLAN$_{ZCP}$',
    'adj_gin_cate': 'FLAN$_{CATE}$',
    'adj_gin_arch2vec': 'FLAN$_{Arch2Vec}$',
}


# Create a single figure with three subplots
fig, axes = plt.subplots(1, 3, figsize=(21, 5))

# Make a graphs directory
if not os.path.exists("graphs"):
    os.mkdir("graphs")

# Loop through each space
for idx, space in enumerate(spaces_to_analyze):
    ax = axes[idx]
    print(space)

    # Set plot limits
    ax.set_xlim(pltlims[space]['x'])
    ax.set_ylim(pltlims[space]['y'])

    # Loop through each experiment folder
    for exp, suffix in experiments.items():
        # List all files in the experiment folder
        all_files = os.listdir("./" + exp)
        
        # Find the file that contains the string {space}_samp_eff.csv
        file_name = next((f for f in all_files if f"{space}_samp_eff.csv" in f), None)

        if file_name:
            file_path = os.path.join("./" + exp, file_name)
            df = pd.read_csv(file_path)
        representation_list = None
        if exp == "table3":
            representation_list = exp3_representations[space]
        elif exp == "table4":
            representation_list = exp4_representations[space]

        if representation_list is not None:
            for representation_info in representation_list:
                if isinstance(representation_info, tuple):
                    representation, key_range = representation_info
                else:
                    representation, key_range = representation_info, (None, None)

                mapped_representation = representation_map.get(representation, representation)
                subset = df[df['representation'] == representation]

                # Filter the subset based on the key range
                if key_range[0] is not None:
                    subset = subset[subset['key'] >= key_range[0]]
                if key_range[1] is not None:
                    subset = subset[subset['key'] <= key_range[1]]

                ax.plot(subset['key'], subset['kdt'], label=f"{mapped_representation}{suffix}", marker='o', linewidth=2)

    # Plot tagates_eff data on the current subplot with thicker line
    # ax.plot(list(tagates_eff[space].keys()), list(tagates_eff[space].values()), label="TA-GATES", marker='v', linestyle='dashed', linewidth=2)
    
    if tagates_eff.get(space):
        ax.plot(list(tagates_eff[space].keys()), list(tagates_eff[space].values()), label="TA-GATES", marker='v', linestyle='dashed', linewidth=2)
    
    # Plot gcn_eff data on the current subplot with thicker line
    ax.plot(list(gcn_eff[space].keys()), list(gcn_eff[space].values()), label="GCN", marker='v', linestyle='dashed', linewidth=2)
    # Plot multipredict_eff data on the current subplot with thicker line
    ax.plot(list(multipredict_eff[space].keys()), list(multipredict_eff[space].values()), label="MultiPredict", marker='v', linestyle='dashed', linewidth=2)


    # Beautify the plot
    ax.set_title(f"Predictor Sample Efficiency for {space_map[space]}")
    ax.set_xlabel("Number of Training Samples")
    ax.set_xscale('log', base=2)
    # ax.set_yscale('log')
    ax.set_ylabel("Kendall Tau")
    ax.legend(loc='upper left', fontsize=8)  # This ensures each subplot has its own legend
    ax.grid(True, which="both", ls="--", c='0.7')  # Add a grid for better readability
    ax.get_xaxis().set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))
    ax.get_xaxis().set_minor_formatter(ticker.StrMethodFormatter("{x:.0f}"))
    
    ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)

# Adjust the layout to make it tight
plt.tight_layout()

# Save the entire figure containing all three subplots
plt.savefig("graphs/combined_spaces_subset.png")
plt.savefig("graphs/combined_spaces_subset.pdf")

print("Graphs saved.")


# import pandas as pd
# import matplotlib.pyplot as plt
# import os

# # Enable LaTeX interpretation in matplotlib
# plt.rcParams['text.usetex'] = False
# plt.rcParams['mathtext.fontset'] = 'cm'


# # Define the tagates_eff dictionary

# tagates_eff = {
#     'nb101': dict(zip([72, 364, 729, 3645, 7290], [0.6686, 0.7744, 0.7839, 0.8133, 0.8217])),
#     'nb201': dict(zip([7, 39, 78, 390, 781], [0.5382, 0.6707, 0.7731, 0.8660, 0.8890])),
#     'ENAS': dict(zip([25, 50, 125, 250, 500], [0.3458, 0.4407, 0.5485, 0.6324, 0.6683])),
# }
# # List of spaces to analyze
# spaces_to_analyze = ['nb101', 'nb201', 'ENAS']  # Add other spaces to this list as needed

# # Experiment folders and their corresponding suffixes
# experiments = {
#     "exp3": "",
#     "exp4": "$^{T}$",
#     "exp5": "$^{UT}$"
# }

# space_map = {'nb101': "NASBench-101", "nb201": "NASBench-201", "ENAS": "ENAS"}

# # Special representations for exp5
# exp5_representations = ['adj_gin_cate', 'adj_gin_arch2vec']

# representation_map = {
#     'adj_gin': 'FLAN',
#     'adj_gin_zcp': 'FLAN$_{ZCP}$',
#     'adj_gin_cate': 'FLAN$_{CATE}$',
#     'adj_gin_arch2vec': 'FLAN$_{Arch2Vec}$',
# }


# # Make a graphs directory
# if not os.path.exists("graphs"):
#     os.mkdir("graphs")

# # Loop through each space
# for space in spaces_to_analyze:
#     plt.figure()

#     # Plot tagates_eff data
#     plt.plot(list(tagates_eff[space].keys()), list(tagates_eff[space].values()), label="tagates_eff", marker='o')

#     # Loop through each experiment folder
#     for exp, suffix in experiments.items():
#         # List all files in the experiment folder
#         all_files = os.listdir("correlation_results/" + exp)
        
#         # Find the file that contains the string {space}_samp_eff.csv
#         file_name = next((f for f in all_files if f"{space}_samp_eff.csv" in f), None)

#         if file_name:
#             file_path = os.path.join("correlation_results/" + exp, file_name)
#             df = pd.read_csv(file_path)

#             # Filter representations for exp5
#             if exp == "exp5":
#                 df = df[df['representation'].isin(exp5_representations)]

#             # Plot for each representation
#             representations = df['representation'].unique()
#             for representation in representations:
#                 mapped_representation = representation_map.get(representation, representation)
#                 subset = df[df['representation'] == representation]
#                 plt.plot(subset['key'], subset['kdt'], label=f"{mapped_representation}{suffix}", marker='x')

#     # Beautify the plot
#     plt.title(f"Results for {space_map[space]}")
#     plt.xlabel("Number of Training Samples")
#     plt.xscale('log')
#     plt.ylabel("Kendall Tau")
#     plt.legend()
#     plt.tight_layout()

#     # Save the plot
#     plt.savefig(f"graphs/tagates_comp_{space}.png")
#     plt.savefig(f"graphs/tagates_comp_{space}.pdf")
#     plt.close()

# print("Graphs saved.")
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import matplotlib.ticker as ticker
from matplotlib.ticker import FormatStrFormatter
from matplotlib.lines import Line2D
import sys
import numpy as np
sys.path.append("./../..")
from nas_embedding_suite.nb101_ss import NASBench101
from nas_embedding_suite.nds_ss import NDS
nb101_embgen = NASBench101(normalize_zcp=True, log_synflow=True)
nds_embgen = NDS()
# Set a consistent color palette
sns.set_palette("tab10")
# Enable LaTeX interpretation in matplotlib
plt.rcParams['text.usetex'] = False
plt.rcParams['mathtext.fontset'] = 'cm'
fsize = 22
plt.rcParams['font.size'] = 16  # Increase font size
space_map = {'nb101': "NASBench-101", "nb201": "NASBench-201", "ENAS_fix-w-d": "ENAS$_{FixWD}$", "DARTS": "DARTS"}
if True:
    exp6_representations = {"nb101": ['adj_gin', 'adj_gin_a2vcatezcp'],
                            "nb201": ['adj_gin', 'adj_gin_a2vcatezcp'],
                            "ENAS_fix-w-d": ['adj_gin', 'adj_gin_a2vcatezcp'],
                            "DARTS": ['adj_gin', 'adj_gin_a2vcatezcp']
                            }
    exp7_representations = {"nb101": ['adj_gin', 'adj_gin_a2vcatezcp'],
                            "nb201": ['adj_gin', 'adj_gin_a2vcatezcp'],
                            "ENAS_fix-w-d": ['adj_gin', 'adj_gin_a2vcatezcp'],
                            "DARTS": ['adj_gin', 'adj_gin_a2vcatezcp']
                            }
    exp8_representations = {"nb101": ['adj_gin_zcp'],
                            "nb201": ['adj_gin_cate'],
                            "ENAS_fix-w-d": [],
                            "DARTS": []
                            }
    representation_map = {
        'adj_gin': '',
        'adj_gin_zcp': '$_{ZCP}$',
        'adj_gin_cate': '$_{CATE}$',
        'adj_gin_arch2vec': '$_{Arch2Vec}$',
        'adj_gin_a2vcatezcp': '$_{CAZ}$',
    }
    lim_map = {
        "nb201": {"y": (0.90, 0.918), "x": (8, 32)},
        "ENAS_fix-w-d": {"y": (0.93, 0.943), "x": (4, 64)},
        "nb101": {"y": (0.925, 0.945), "x": (4, 32)},
        "DARTS": {"y": (0.93, 0.943), "x": (4, 128)},
    }
    if not os.path.exists("search_graphs"):
        os.makedirs("search_graphs")
    experiments = {
        "exp6": "",
        "exp7": "$^{T}$"
    }
    reference_csv_files = [f for f in os.listdir(next(iter(experiments))) if f.endswith("_search_eff.csv")]
    inv_tf = nb101_embgen.min_max_scaler.inverse_transform
    ndsinv_tf = nds_embgen.minmax_sc['ENAS_fix-w-d'].inverse_transform
    plt.cla()
    spaces_to_analyze = ['nb101', 'nb201', 'DARTS', 'ENAS_fix-w-d']  # Modify this list as needed
    plt.clf()
    legend_handles = {}
    legend_artists = []
    legend_labels = []
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for idx, space in enumerate(spaces_to_analyze):
        ax = axes[idx]
        ax.set_xlim(lim_map[space]['x'])
        ax.set_ylim(lim_map[space]['y'])
        for exp, suffix in experiments.items():
            file_name = f"{space}_search_eff.csv"
            file_path = os.path.join(exp, file_name)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df = df.apply(lambda col: col.str.strip() if col.dtype == 'object' else col)
                if exp == "exp6":
                    reps_to_plot = exp6_representations.get(space, [])
                if exp == "exp7":
                    reps_to_plot = exp7_representations.get(space, [])
                if exp == "exp8":
                    reps_to_plot = exp8_representations.get(space, [])
                reps_to_plot = [r for r in reps_to_plot if r in df['representation'].unique()]
                for representation in reps_to_plot:
                    mapped_representation = representation_map.get(representation, representation)
                    subset = df[df['representation'] == representation]
                    legend_label = f"FLAN{suffix}{mapped_representation}"
                    proxy_artist = Line2D([0], [0], color='black', marker='o', linestyle='-', label=legend_label)
                    legend_artists.append(proxy_artist)
                    legend_labels.append(legend_label)
                    if space in ['nb101', 'ENAS_fix-w-d', 'DARTS']:
                        old_subs = subset.copy()
                        if space == 'nb101':
                            subset['av_best_acc'] = inv_tf(subset[['av_best_acc']]).reshape(-1)
                        else:
                            subset['av_best_acc'] = ndsinv_tf(subset[['av_best_acc']]).reshape(-1)
                        inverse_transformed_std_devs = []
                        for index, row in subset.iterrows():
                            synthetic_points = np.random.normal(loc=old_subs[old_subs.index==index]['av_best_acc'], scale=row['best_acc_std'], size=10000)
                            if space == 'nb101':
                                inverse_transformed_points = inv_tf(synthetic_points.reshape(-1, 1)).reshape(-1)
                            else:
                                inverse_transformed_points = ndsinv_tf(synthetic_points.reshape(-1, 1)).reshape(-1)
                            inverse_transformed_std_dev = np.std(inverse_transformed_points)
                            inverse_transformed_std_devs.append(inverse_transformed_std_dev)
                        subset['best_acc_std'] = inverse_transformed_std_devs
                    ls = '--' if exp=='exp6' else '-'
                    if space in ['ENAS_fix-w-d', 'DARTS']:
                        subset['best_acc_std'] = subset['best_acc_std']/100.
                        subset['av_best_acc'] = subset['av_best_acc']/100.
                    handle, = ax.plot(subset['num_samps'], subset['av_best_acc'], label=f"FLAN{suffix}{mapped_representation}", marker='P', markersize=10, linewidth=3, linestyle=ls)
                    stmax = 0.945 if space=='nb101' else 1
                    stmax = 0.94 if space=='ENAS_fix-w-d' else stmax
                    if idx==1:
                        ax.legend(loc='center', ncol=4, fontsize=fsize, bbox_to_anchor=(1.2, 1.3))
                    ax.fill_between(subset['num_samps'],
                                    subset['av_best_acc'] - subset['best_acc_std'], 
                                    [min(stmax, x) for x in subset['av_best_acc'] + subset['best_acc_std']],
                                    alpha=0.1)
                    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.set_title(f"{space_map[space]}", fontsize=fsize)
        ax.set_xscale('log', basex=2)
        ax.get_xaxis().set_minor_formatter(ticker.StrMethodFormatter("{x:.0f}"))
        ax.get_xaxis().set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))
        ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
        if idx==0:
            ax.set_ylabel("Average Best Accuracy", fontsize=fsize, labelpad=15)
        ax.grid(True, which="both", ls="--", c='0.7')  # Add a grid for better readability
    plt.subplots_adjust(bottom=0.16,top=0.70, left=0.07, right=0.99, wspace=0.3)  # Adjust as needed
    fig.text(0.5, 0.03, 'Number of Trained Networks', ha='center', va='center', fontsize=fsize)
    plt.savefig("search_graphs/combined_spaces_dash_rbt.png")
    plt.savefig("search_graphs/combined_spaces_dash_rbt.pdf")
    print("Graphs saved in 'search_graphs' folder.")
        # ax.get_xaxis().set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))
plt.subplots_adjust(top=0.75, wspace=0.3)  # Adjust the top value as needed
    # fig.text(0.5, 0.023, 'Number of Trained Networks', ha='center', va='center', fontsize=fsize)
        # ax.text(0.98,.05,f"{space_map[space]}", fontsize=fsize,
        #     horizontalalignment='right', bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3'),
        #     transform=ax.transAxes)
    # exp6_representations = {"nb101": ['adj_gin', 'adj_gin_zcp', 'adj_gin_cate', 'adj_gin_arch2vec', 'adj_gin_a2vcatezcp'],
    #                         "nb201": ['adj_gin', 'adj_gin_zcp', 'adj_gin_cate', 'adj_gin_arch2vec', 'adj_gin_a2vcatezcp'],
    #                         "ENAS_fix-w-d": ['adj_gin', 'adj_gin_zcp', 'adj_gin_cate', 'adj_gin_arch2vec', 'adj_gin_a2vcatezcp']
    #                         }
        # ax.set_xlabel("Number of Trained Networks")
    # exp7_representations = {"nb101": ['adj_gin', 'adj_gin_zcp', 'adj_gin_cate', 'adj_gin_arch2vec', 'adj_gin_a2vcatezcp'],
    #                         "nb201": ['adj_gin', 'adj_gin_zcp', 'adj_gin_cate', 'adj_gin_arch2vec', 'adj_gin_a2vcatezcp'],
    #                         "ENAS_fix-w-d": ['adj_gin', 'adj_gin_zcp', 'adj_gin_cate', 'adj_gin_arch2vec', 'adj_gin_a2vcatezcp'],
    #                         }
    # exp8_representations = {"nb101": ['adj_gin_zcp'],
    #                         "nb201": ['adj_gin_cate'],
    #                         "ENAS_fix-w-d": []
    #                         }
# unif_color = 'dash' # 'color', false, 'dash'
# if unif_color == 'color':
#     import pandas as pd
#     import matplotlib.pyplot as plt
#     import os
#     import seaborn as sns
#     import matplotlib.ticker as ticker
#     import sys
#     import numpy as np
#     sys.path.append("./../..")
#     from nas_embedding_suite.nb101_ss import NASBench101
#     nb101_embgen = NASBench101(normalize_zcp=True, log_synflow=True)

#     # Set a consistent color palette
#     sns.set_palette("tab10")

#     # Enable LaTeX interpretation in matplotlib
#     plt.rcParams['text.usetex'] = False
#     plt.rcParams['mathtext.fontset'] = 'cm'
#     plt.rcParams['font.size'] = 12  # Increase font size

#     space_map = {'nb101': "NASBench-101", "nb201": "NASBench-201", "ENAS_fix-w-d": "ENAS$_{FixWD}$"}

#     # Define the experiment folders and their corresponding suffixes
#     experiments = {
#         "exp7": "$^{T}$",
#         "exp6": "",
#         "exp8": "$^{UT}$"
#     }
#     exp6_representations = {"nb101": ['adj_gin', 'adj_gin_zcp', 'adj_gin_cate', 'adj_gin_arch2vec', 'adj_gin_a2vcatezcp'],
#                             "nb201": ['adj_gin', 'adj_gin_zcp', 'adj_gin_cate', 'adj_gin_arch2vec', 'adj_gin_a2vcatezcp'],
#                             "ENAS_fix-w-d": ['adj_gin', 'adj_gin_zcp', 'adj_gin_cate', 'adj_gin_arch2vec', 'adj_gin_a2vcatezcp']
#                             }
#     exp7_representations = {"nb101": ['adj_gin', 'adj_gin_zcp', 'adj_gin_cate', 'adj_gin_arch2vec', 'adj_gin_a2vcatezcp'],
#                             "nb201": ['adj_gin', 'adj_gin_zcp', 'adj_gin_cate', 'adj_gin_arch2vec', 'adj_gin_a2vcatezcp'],
#                             "ENAS_fix-w-d": ['adj_gin', 'adj_gin_zcp', 'adj_gin_cate', 'adj_gin_arch2vec', 'adj_gin_a2vcatezcp'],
#                             }
#     exp8_representations = {"nb101": ['adj_gin_zcp'],
#                             "nb201": ['adj_gin_cate'],
#                             "ENAS_fix-w-d": []
#                             }
#     representation_map = {
#         'adj_gin': '',
#         'adj_gin_zcp': '$_{ZCP}$',
#         'adj_gin_cate': '$_{CATE}$',
#         'adj_gin_arch2vec': '$_{Arch2Vec}$',
#         'adj_gin_a2vcatezcp': '$_{CAZ}$',
#     }

#     # Define different markers for each representation
#     markers = {
#         'adj_gin': 'o',
#         'adj_gin_zcp': 's',
#         'adj_gin_cate': '^',
#         'adj_gin_arch2vec': 'D',
#         'adj_gin_a2vcatezcp': 'P',
#     }
#     colors = {
#         "exp6": "blue",
#         "exp7": "green",
#         "exp8": "red"
#     }
#     lim_map = {
#         "nb201": {"y": (0.875, 0.925), "x": (4, 64)},
#         "ENAS_fix-w-d": {"y": (0.875, 1.01), "x": (4, 64)},
#         "nb101": {"y": (0.925, 0.950), "x": (4, 32)},
#     }

#     # Create the search_graphs directory if it doesn't exist
#     if not os.path.exists("search_graphs"):
#         os.makedirs("search_graphs")

#     # Get the list of csv files from the first experiment folder as a reference
#     reference_csv_files = [f for f in os.listdir(next(iter(experiments))) if f.endswith("_search_eff.csv")]


#     # List of spaces to analyze
#     spaces_to_analyze = ['nb101', 'nb201', 'ENAS_fix-w-d']  # Modify this list as needed

#     # Create a single figure with three subplots
#     fig, axes = plt.subplots(1, 3, figsize=(25, 7))

#     inv_tf = nb101_embgen.min_max_scaler.inverse_transform
#     # Loop through each space
#     for idx, space in enumerate(spaces_to_analyze):
#         ax = axes[idx]
#         # ... [same as before]
#         # Loop through each experiment folder
#         for exp, suffix in experiments.items():
#             file_name = f"{space}_search_eff.csv"
#             file_path = os.path.join(exp, file_name)

#             # Check if the file exists
#             if os.path.exists(file_path):
#                 df = pd.read_csv(file_path)
#                 df = df.apply(lambda col: col.str.strip() if col.dtype == 'object' else col)

#                 # Get the representations to be plotted for the current experiment and space
#                 reps_to_plot = []
#                 if exp == "exp6":
#                     reps_to_plot = exp6_representations.get(space, [])
#                 elif exp == "exp7":
#                     reps_to_plot = exp7_representations.get(space, [])
#                 elif exp == "exp8":
#                     reps_to_plot = exp8_representations.get(space, [])

#                 reps_to_plot = [r for r in reps_to_plot if r in df['representation'].unique()]

#                 # Plot for each specified representation
#                 for representation in reps_to_plot:
#                     mapped_representation = representation_map.get(representation, representation)
#                     subset = df[df['representation'] == representation]
#                     if space == 'nb101':
#                         subset['av_best_acc'] = inv_tf(subset[['av_best_acc']]).reshape(-1)
#                         inverse_transformed_std_devs = []
#                         for index, row in subset.iterrows():
#                             synthetic_points = np.random.normal(loc=row['av_best_acc'], scale=row['best_acc_std'], size=10000)
#                             inverse_transformed_points = inv_tf(synthetic_points.reshape(-1, 1)).reshape(-1)
#                             inverse_transformed_std_dev = np.std(inverse_transformed_points)
#                             inverse_transformed_std_devs.append(inverse_transformed_std_dev)
#                         subset['best_acc_std'] = inverse_transformed_std_devs

#                     # Use the color and marker specified for each experiment and representation respectively
#                     ax.plot(subset['num_samps'], subset['av_best_acc'], label=f"FLAN{suffix}{mapped_representation}",
#                             color=colors[exp], marker=markers[representation], markersize=4)
#                     # ax.fill_between(subset['num_samps'],
#                     #                 subset['av_best_acc'] - subset['best_acc_std'],
#                     #                 [min(1, x) for x in subset['av_best_acc'] + subset['best_acc_std']],
#                     #                 color=colors[exp], alpha=0.1)


#         # Beautify the plot
#         ax.set_title(f"NAS Search on {space_map[space]}")
#         ax.set_xlabel("Number of Samples")
#         ax.set_xscale('log', basex=2)
#         ax.get_xaxis().set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))
#         ax.get_xaxis().set_minor_formatter(ticker.StrMethodFormatter("{x:.0f}"))
#         ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
#         ax.set_ylabel("Average Best Accuracy")
#         ax.legend(loc='lower right', fontsize=8)  # This ensures each subplot has its own legend
#         ax.grid(True, which="both", ls="--", c='0.7')  # Add a grid for better readability

#     # Adjust the layout to make it tight
#     plt.tight_layout()

#     # Save the entire figure containing all three subplots
#     plt.savefig("search_graphs/combined_spaces_unifc.png")
#     plt.savefig("search_graphs/combined_spaces_unifc.pdf")

#     print("Graphs saved in 'search_graphs' folder.")
# elif unif_color=='dash':
#     import pandas as pd
#     import matplotlib.pyplot as plt
#     import os
#     import seaborn as sns
#     import matplotlib.ticker as ticker
#     from matplotlib.lines import Line2D
#     import sys
#     import numpy as np
#     sys.path.append("./../..")
#     from nas_embedding_suite.nb101_ss import NASBench101
#     from nas_embedding_suite.nds_ss import NDS
#     nb101_embgen = NASBench101(normalize_zcp=True, log_synflow=True)
#     nds_embgen = NDS()

#     # Set a consistent color palette
#     sns.set_palette("tab10")

#     # Enable LaTeX interpretation in matplotlib
#     plt.rcParams['text.usetex'] = False
#     plt.rcParams['mathtext.fontset'] = 'cm'
#     plt.rcParams['font.size'] = 18  # Increase font size

#     space_map = {'nb101': "NASBench-101", "nb201": "NASBench-201", "ENAS_fix-w-d": "ENAS$_{FixWD}$", "DARTS": "DARTS"}

#     # exp6_representations = {"nb101": ['adj_gin', 'adj_gin_zcp', 'adj_gin_cate', 'adj_gin_arch2vec', 'adj_gin_a2vcatezcp'],
#     #                         "nb201": ['adj_gin', 'adj_gin_zcp', 'adj_gin_cate', 'adj_gin_arch2vec', 'adj_gin_a2vcatezcp'],
#     #                         "ENAS_fix-w-d": ['adj_gin', 'adj_gin_zcp', 'adj_gin_cate', 'adj_gin_arch2vec', 'adj_gin_a2vcatezcp']
#     #                         }
#     # exp7_representations = {"nb101": ['adj_gin', 'adj_gin_zcp', 'adj_gin_cate', 'adj_gin_arch2vec', 'adj_gin_a2vcatezcp'],
#     #                         "nb201": ['adj_gin', 'adj_gin_zcp', 'adj_gin_cate', 'adj_gin_arch2vec', 'adj_gin_a2vcatezcp'],
#     #                         "ENAS_fix-w-d": ['adj_gin', 'adj_gin_zcp', 'adj_gin_cate', 'adj_gin_arch2vec', 'adj_gin_a2vcatezcp'],
#     #                         }
#     # exp8_representations = {"nb101": ['adj_gin_zcp'],
#     #                         "nb201": ['adj_gin_cate'],
#     #                         "ENAS_fix-w-d": []
#     #                         }
# if True:
#     exp6_representations = {"nb101": ['adj_gin', 'adj_gin_a2vcatezcp'],
#                             "nb201": ['adj_gin', 'adj_gin_a2vcatezcp'],
#                             "ENAS_fix-w-d": ['adj_gin', 'adj_gin_a2vcatezcp'],
#                             "DARTS": ['adj_gin', 'adj_gin_a2vcatezcp']
#                             }
#     exp7_representations = {"nb101": ['adj_gin', 'adj_gin_a2vcatezcp'],
#                             "nb201": ['adj_gin', 'adj_gin_a2vcatezcp'],
#                             "ENAS_fix-w-d": ['adj_gin', 'adj_gin_a2vcatezcp'],
#                             "DARTS": ['adj_gin', 'adj_gin_a2vcatezcp']
#                             }
#     exp8_representations = {"nb101": ['adj_gin_zcp'],
#                             "nb201": ['adj_gin_cate'],
#                             "ENAS_fix-w-d": [],
#                             "DARTS": []
#                             }
#     representation_map = {
#         'adj_gin': '',
#         'adj_gin_zcp': '$_{ZCP}$',
#         'adj_gin_cate': '$_{CATE}$',
#         'adj_gin_arch2vec': '$_{Arch2Vec}$',
#         'adj_gin_a2vcatezcp': '$_{CAZ}$',
#     }

#     lim_map = {
#         "nb201": {"y": (0.90, 0.92), "x": (8, 64)},
#         "ENAS_fix-w-d": {"y": (0.93, 0.943), "x": (8, 64)},
#         "nb101": {"y": (0.93, 0.95), "x": (8, 64)},
#         "DARTS": {"y": (0.93, 0.943), "x": (8, 128)},
#     }
#     if not os.path.exists("search_graphs"):
#         os.makedirs("search_graphs")
#     experiments = {
#         "exp6": "",
#         "exp7": "$^{T}$"
#     }
#     reference_csv_files = [f for f in os.listdir(next(iter(experiments))) if f.endswith("_search_eff.csv")]
#     inv_tf = nb101_embgen.min_max_scaler.inverse_transform
#     ndsinv_tf = nds_embgen.minmax_sc['ENAS_fix-w-d'].inverse_transform
#     plt.cla()
#     spaces_to_analyze = ['nb101', 'nb201', 'DARTS', 'ENAS_fix-w-d']  # Modify this list as needed
#     plt.clf()
#     legend_handles = {}
#     legend_artists = []
#     legend_labels = []
#     fig, axes = plt.subplots(1, 4, figsize=(20, 5))
#     for idx, space in enumerate(spaces_to_analyze):
#         ax = axes[idx]
#         ax.set_xlim(lim_map[space]['x'])
#         ax.set_ylim(lim_map[space]['y'])
#         for exp, suffix in experiments.items():
#             file_name = f"{space}_search_eff.csv"
#             file_path = os.path.join(exp, file_name)
#             if os.path.exists(file_path):
#                 df = pd.read_csv(file_path)
#                 df = df.apply(lambda col: col.str.strip() if col.dtype == 'object' else col)
#                 if exp == "exp6":
#                     reps_to_plot = exp6_representations.get(space, [])
#                 if exp == "exp7":
#                     reps_to_plot = exp7_representations.get(space, [])
#                 if exp == "exp8":
#                     reps_to_plot = exp8_representations.get(space, [])
#                 reps_to_plot = [r for r in reps_to_plot if r in df['representation'].unique()]
#                 for representation in reps_to_plot:
#                     mapped_representation = representation_map.get(representation, representation)
#                     subset = df[df['representation'] == representation]
#                     legend_label = f"FLAN{suffix}{mapped_representation}"
#                     proxy_artist = Line2D([0], [0], color='black', marker='o', linestyle='-', label=legend_label)
#                     legend_artists.append(proxy_artist)
#                     legend_labels.append(legend_label)
#                     if space in ['nb101', 'ENAS_fix-w-d', 'DARTS']:
#                         old_subs = subset.copy()
#                         if space == 'nb101':
#                             subset['av_best_acc'] = inv_tf(subset[['av_best_acc']]).reshape(-1)
#                         else:
#                             subset['av_best_acc'] = ndsinv_tf(subset[['av_best_acc']]).reshape(-1)
#                         inverse_transformed_std_devs = []
#                         for index, row in subset.iterrows():
#                             synthetic_points = np.random.normal(loc=old_subs[old_subs.index==index]['av_best_acc'], scale=row['best_acc_std'], size=10000)
#                             if space == 'nb101':
#                                 inverse_transformed_points = inv_tf(synthetic_points.reshape(-1, 1)).reshape(-1)
#                             else:
#                                 inverse_transformed_points = ndsinv_tf(synthetic_points.reshape(-1, 1)).reshape(-1)
#                             inverse_transformed_std_dev = np.std(inverse_transformed_points)
#                             inverse_transformed_std_devs.append(inverse_transformed_std_dev)
#                         subset['best_acc_std'] = inverse_transformed_std_devs
#                     ls = '--' if exp=='exp6' else '-'
#                     if space in ['ENAS_fix-w-d', 'DARTS']:
#                         subset['best_acc_std'] = subset['best_acc_std']/100.
#                         subset['av_best_acc'] = subset['av_best_acc']/100.
#                     handle, = ax.plot(subset['num_samps'], subset['av_best_acc'], label=f"FLAN{suffix}{mapped_representation}", marker='o', markersize=5, linewidth=1, linestyle=ls)
#                     if idx==1:
#                         ax.legend(loc='upper center', ncol=4, fontsize=fsize, bbox_to_anchor=(0.5, 1.2))  # This ensures each subplot has its own legend
#                     stmax = 0.945 if space=='nb101' else 1
#                     stmax = 0.94 if space=='ENAS_fix-w-d' else stmax
#                     ax.fill_between(subset['num_samps'],
#                                     subset['av_best_acc'] - subset['best_acc_std'], 
#                                     [min(stmax, x) for x in subset['av_best_acc'] + subset['best_acc_std']],
#                                     alpha=0.05)
#         ax.text(0.98,.05,f"{space_map[space]}", fontsize=fsize,
#             horizontalalignment='right', bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3'),
#             transform=ax.transAxes)
#         ax.set_xlabel("Number of Trained Networks")
#         ax.set_xscale('log', basex=2)
#         ax.get_xaxis().set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))
#         ax.get_xaxis().set_minor_formatter(ticker.StrMethodFormatter("{x:.0f}"))
#         ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
#         ax.set_ylabel("Average Best Accuracy")
#         ax.grid(True, which="both", ls="--", c='0.7')  # Add a grid for better readability
#     plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space at the bottom for the unified legend
#     plt.savefig("search_graphs/combined_spaces_dash_rbt.png")
#     plt.savefig("search_graphs/combined_spaces_dash_rbt.pdf")
#     print("Graphs saved in 'search_graphs' folder.")

# else:
#     import pandas as pd
#     import matplotlib.pyplot as plt
#     import os
#     import seaborn as sns
#     import matplotlib.ticker as ticker
#     import sys
#     import numpy as np
#     sys.path.append("./../..")
#     from nas_embedding_suite.nb101_ss import NASBench101
#     from nas_embedding_suite.nds_ss import NDS
#     nb101_embgen = NASBench101(normalize_zcp=True, log_synflow=True)
#     nds_embgen = NDS()

#     # Set a consistent color palette
#     sns.set_palette("tab10")

#     # Enable LaTeX interpretation in matplotlib
#     plt.rcParams['text.usetex'] = False
#     plt.rcParams['mathtext.fontset'] = 'cm'
#     plt.rcParams['font.size'] = 12  # Increase font size

#     space_map = {'nb101': "NASBench-101", "nb201": "NASBench-201", "ENAS_fix-w-d": "ENAS$_{FixWD}$"}

#     # Define the experiment folders and their corresponding suffixes
#     experiments = {
#         "exp7": "$^{T}$",
#         "exp6": "",
#         "exp8": "$^{UT}$"
#     }
#     exp6_representations = {"nb101": ['adj_gin', 'adj_gin_zcp', 'adj_gin_cate', 'adj_gin_arch2vec', 'adj_gin_a2vcatezcp'],
#                             "nb201": ['adj_gin', 'adj_gin_zcp', 'adj_gin_cate', 'adj_gin_arch2vec', 'adj_gin_a2vcatezcp'],
#                             "ENAS_fix-w-d": ['adj_gin', 'adj_gin_zcp', 'adj_gin_cate', 'adj_gin_arch2vec', 'adj_gin_a2vcatezcp']
#                             }
#     exp7_representations = {"nb101": ['adj_gin', 'adj_gin_zcp', 'adj_gin_cate', 'adj_gin_arch2vec', 'adj_gin_a2vcatezcp'],
#                             "nb201": ['adj_gin', 'adj_gin_zcp', 'adj_gin_cate', 'adj_gin_arch2vec', 'adj_gin_a2vcatezcp'],
#                             "ENAS_fix-w-d": ['adj_gin', 'adj_gin_zcp', 'adj_gin_cate', 'adj_gin_arch2vec', 'adj_gin_a2vcatezcp'],
#                             }
#     exp8_representations = {"nb101": ['adj_gin_zcp'],
#                             "nb201": ['adj_gin_cate'],
#                             "ENAS_fix-w-d": []
#                             }
#     representation_map = {
#         'adj_gin': '',
#         'adj_gin_zcp': '$_{ZCP}$',
#         'adj_gin_cate': '$_{CATE}$',
#         'adj_gin_arch2vec': '$_{Arch2Vec}$',
#         'adj_gin_a2vcatezcp': '$_{CAZ}$',
#     }

#     lim_map = {
#         "nb201": {"y": (0.875, 0.925), "x": (4, 64)},
#         "ENAS_fix-w-d": {"y": (0.875, 1.01), "x": (8, 64)},
#         "nb101": {"y": (0.925, 0.950), "x": (8, 64)},
#     }

#     # Create the search_graphs directory if it doesn't exist
#     if not os.path.exists("search_graphs"):
#         os.makedirs("search_graphs")

#     # Get the list of csv files from the first experiment folder as a reference
#     reference_csv_files = [f for f in os.listdir(next(iter(experiments))) if f.endswith("_search_eff.csv")]


#     # List of spaces to analyze
#     spaces_to_analyze = ['nb101', 'nb201', 'ENAS_fix-w-d']  # Modify this list as needed

#     # Create a single figure with three subplots
#     fig, axes = plt.subplots(1, 3, figsize=(21, 5))

#     inv_tf = nb101_embgen.min_max_scaler.inverse_transform
#     ndsinv_tf = nds_embgen.minmax_sc['ENAS'].inverse_transform
#     # Loop through each space
#     for idx, space in enumerate(spaces_to_analyze):
#         ax = axes[idx]

#         # Set plot limits
#         ax.set_xlim(lim_map[space]['x'])
#         ax.set_ylim(lim_map[space]['y'])

#         # Loop through each experiment folder
#         for exp, suffix in experiments.items():
#             file_name = f"{space}_search_eff.csv"
#             file_path = os.path.join(exp, file_name)

#             # Check if the file exists
#             if os.path.exists(file_path):
#                 df = pd.read_csv(file_path)
#                     # Stripping spaces from string columns
#                 df = df.apply(lambda col: col.str.strip() if col.dtype == 'object' else col)
                
#                 # Get the representations to be plotted for the current experiment and space
#                 if exp == "exp6":
#                     reps_to_plot = exp6_representations.get(space, [])
#                 if exp == "exp7":
#                     reps_to_plot = exp7_representations.get(space, [])
#                 if exp == "exp8":
#                     reps_to_plot = exp8_representations.get(space, [])
#                 reps_to_plot = [r for r in reps_to_plot if r in df['representation'].unique()]

#                 # Plot for each specified representation
#                 for representation in reps_to_plot:
#                     mapped_representation = representation_map.get(representation, representation)
#                     subset = df[df['representation'] == representation]
#                     # Apply the inverse transform if space is 'nb101'
#                     invt_f = inv_tf if space=='nb101' else ndsinv_tf
#                     if space in ['nb101', 'ENAS_fix-w-d', 'DARTS']:
#                         subset['av_best_acc'] = invt_f(subset[['av_best_acc']]).reshape(-1)
#                         inverse_transformed_std_devs = []
#                         for index, row in subset.iterrows():
#                             synthetic_points = np.random.normal(loc=row['av_best_acc'], scale=row['best_acc_std'], size=10000)
#                             inverse_transformed_points = invt_f(synthetic_points.reshape(-1, 1)).reshape(-1)
#                             inverse_transformed_std_dev = np.std(inverse_transformed_points)
#                             inverse_transformed_std_devs.append(inverse_transformed_std_dev)
#                         subset['best_acc_std'] = inverse_transformed_std_devs
#                     ax.plot(subset['num_samps'], subset['av_best_acc'], label=f"FLAN{suffix}{mapped_representation}", marker='o', markersize=4)

#                     # Add shaded error regions
#                     ax.fill_between(subset['num_samps'],
#                                     subset['av_best_acc'] - subset['best_acc_std'],
#                                     [min(1, x) for x in subset['av_best_acc'] + subset['best_acc_std']],
#                                     alpha=0.1)


#         # Beautify the plot
#         ax.set_title(f"NAS Search on {space_map[space]}")
#         ax.set_xlabel("Number of Samples")
#         ax.set_xscale('log', basex=2)
#         ax.get_xaxis().set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))
#         ax.get_xaxis().set_minor_formatter(ticker.StrMethodFormatter("{x:.0f}"))
#         ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
#         ax.set_ylabel("Average Best Accuracy")
#         ax.legend(loc='lower right', fontsize=8)  # This ensures each subplot has its own legend
#         ax.grid(True, which="both", ls="--", c='0.7')  # Add a grid for better readability

#     # Adjust the layout to make it tight
#     plt.tight_layout()

#     # Save the entire figure containing all three subplots
#     plt.savefig("search_graphs/combined_spaces.png")
#     plt.savefig("search_graphs/combined_spaces.pdf")

#     print("Graphs saved in 'search_graphs' folder.")

# # import pandas as pd
# # import matplotlib.pyplot as plt
# # import os
# # import seaborn as sns

# # # Set a consistent color palette
# # sns.set_palette("tab10")

# # # Enable LaTeX interpretation in matplotlib
# # plt.rcParams['text.usetex'] = False
# # plt.rcParams['mathtext.fontset'] = 'cm'
# # plt.rcParams['font.size'] = 6  # Increase font size


# # # Define the experiment folders and their corresponding suffixes
# # experiments = {
# #     "exp7": "$^{T}$",
# #     "exp6": "",
# #     "exp8": "$^{UT}$"
# # }

# # representation_map = {
# #     'adj_gin': 'UGN',
# #     'adj_gin_zcp': 'UGN$_{ZCP}$',
# #     'adj_gin_cate': 'UGN$_{CATE}$',
# #     'adj_gin_arch2vec': 'UGN$_{Arch2Vec}$',
# # }

# # lim_map = {
# #     "nb201": {"y": (0.875, 0.925), "x": (2, 256)},
# #     "ENAS": {"y": (0.875, 0.925), "x": (2, 256)},
# #     "nb101": {"y": (0.8, 0.925), "x": (2, 256)},
# # }

# # # Create the search_graphs directory if it doesn't exist
# # if not os.path.exists("search_graphs"):
# #     os.makedirs("search_graphs")

# # # Get the list of csv files from the first experiment folder as a reference
# # reference_csv_files = [f for f in os.listdir(next(iter(experiments))) if f.endswith("_search_eff.csv")]

# # # Loop through each csv file (each space)
# # for csv_file in reference_csv_files:
# #     space = csv_file.replace("_search_eff.csv", "")
    
# #     # Create a new figure for the space
# #     plt.figure()

# #     # Loop through each experiment folder
# #     for exp, suffix in experiments.items():
# #         file_path = os.path.join(exp, csv_file)
        
# #         # Check if the file exists
# #         if os.path.exists(file_path):
# #             df = pd.read_csv(file_path)

# #             # Plot for each representation
# #             representations = df['representation'].unique()
# #             for representation in representations:
# #                 mapped_representation = representation_map[representation.replace(" ", "")]
# #                 subset = df[df['representation'] == representation]
# #                 plt.plot(subset['num_samps'], subset['av_best_acc'], label=f"{mapped_representation}{suffix}")
                
# #                 # Add shaded error regions
# #                 plt.fill_between(subset['num_samps'], 
# #                                 subset['av_best_acc'] - subset['best_acc_std'], 
# #                                 [min(1, x) for x in subset['av_best_acc'] + subset['best_acc_std']], 
# #                                 alpha=0.1)

# #     # Beautify the plot
# #     plt.title(f"Results for {space}")
# #     plt.xlabel("Number of Samples")
# #     # plt.xlim([4, 64])
# #     plt.ylim([0.8, 1])
# #     # plt.yscale("log")
# #     plt.xscale("log", basex=2)
# #     plt.ylabel("Average Best Accuracy")
# #     plt.legend()
# #     plt.tight_layout()
# #     plt.grid(True, which="both", ls="--", c='0.7')  # Add a grid for better readability


# #     # Save the plot
# #     plt.savefig(f"search_graphs/{space}.png")
# #     plt.savefig(f"search_graphs/{space}.pdf")
# #     plt.close()

# # print("Graphs saved in 'search_graphs' folder.")

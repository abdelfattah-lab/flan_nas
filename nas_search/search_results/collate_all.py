mode = 'collate' # generate, collate
if mode == 'generate':
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    import matplotlib.ticker as ticker
    sns.set_theme(context='notebook', style='whitegrid', palette='deep', font_scale=1.2)

    experiments = {
        "allnas_t": "$^{T}$",
        "allnas": "",
    }
    representation_map = {
        'adj_gin': '',
        'adj_gin_zcp': '$_{ZCP}$',
        'adj_gin_cate': '$_{CATE}$',
        'adj_gin_arch2vec': '$_{Arch2Vec}$',
        'adj_gin_a2vcatezcp': '$_{CAZ}$',
    }


    output_folder = "all_graphs"
    os.makedirs(output_folder, exist_ok=True)

    spaces_to_analyze = set()

    for exp in experiments:
        csv_files = [f for f in os.listdir(exp) if f.endswith("_search_eff.csv")]
        for f in csv_files:
            spaces_to_analyze.add(f.rsplit('_search_eff.csv', 1)[0])

    for space in spaces_to_analyze:
        fig = plt.figure(figsize=(10,6))
        
        for exp, suffix in experiments.items():
            file_path = os.path.join(exp, f"{space}_search_eff.csv")
            
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                
                df = df.apply(lambda col: col.str.strip() if col.dtype == 'object' else col)
                df = df.sort_values(by='num_samps')
                # exit(0)
                if max(df['av_best_acc']) > 1:
                    # divide  by 100
                    df['av_best_acc'] = df['av_best_acc'] / 100.
                    df['best_acc_std'] = df['best_acc_std'] / 100.
                representations = df['representation'].unique()
                colors = sns.color_palette("husl", len(representations))
                
                linestyle = '--' if 'allnas' in exp else '-'  # Dashed for 'allnas' and solid for others
                
                for rep, color in zip(representations, colors):
                    mapped_representation = representation_map.get(rep, rep)
                    rep_df = df[df['representation'] == rep]
                    label = f"FLAN{suffix}{mapped_representation}" if suffix else f'FLAN{mapped_representation}'
                    
                    plt.plot(rep_df['num_samps'], rep_df['av_best_acc'], label=label, color=color, linestyle=linestyle)
                    plt.fill_between(rep_df['num_samps'], 
                                    rep_df['av_best_acc'] - rep_df['best_acc_std'], 
                                    [min(1, x) for x in (rep_df['av_best_acc'] + rep_df['best_acc_std'])], alpha=0.1, color=color)
                    
        ax = fig.axes[0]
        ax.get_xaxis().set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))
        ax.get_xaxis().set_minor_formatter(ticker.StrMethodFormatter("{x:.0f}"))
        plt.xscale('log', basex=2)
        plt.xlabel('Number of Trained Models')
        plt.ylabel('Average Best Accuracy')
        plt.title(space)
        plt.legend(loc='lower right')  # Moved the legend to the bottom right
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_folder, f"{space}_g.png"))
        plt.savefig(os.path.join(output_folder, f"{space}_g.pdf"))
        
        plt.close()

    print(f"Graphs saved in '{output_folder}' folder.")
else:
    import os
    import math
    from PIL import Image

    folder_path = 'all_graphs'
    images = [Image.open(os.path.join(folder_path, img)) for img in os.listdir(folder_path) if img.endswith('.png')]

    # If no images were found, exit the script
    if not images:
        print("No images found!")
        exit()

    # Find the size for the resulting image grid
    img_width, img_height = images[0].size  # Assuming all images have the same size
    num_images = len(images)

    # Calculate the grid size
    # cols = rows = int(math.ceil(math.sqrt(num_images)))
    # rows = rows + 1
    cols = 2
    rows = 7

    # Calculate the size of the combined image
    total_width = cols * img_width
    total_height = rows * img_height

    # Create a blank (white) image with the calculated size
    combined_img = Image.new('RGB', (total_width, total_height), 'white')

    # Paste each image into its position in the grid
    for i, img in enumerate(images):
        x = (i % cols) * img_width
        y = (i // cols) * img_height
        combined_img.paste(img, (x, y))

    # Save the resulting image
    combined_img.save('combined_image.png')

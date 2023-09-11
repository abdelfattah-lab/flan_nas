if True:
    flattened_data = abs(np.asarray(diff_mat))
    plt.figure(figsize=(10, 6))
    iqr = np.percentile(flattened_data, 75) - np.percentile(flattened_data, 25)
    bin_width = 2 * iqr * (len(flattened_data) ** (-1/3))
    bin_num = int((flattened_data.max() - flattened_data.min()) / bin_width)
    print(bin_num)
    plt.hist(flattened_data, bins=bin_num, color='green', alpha=0.7)
    plt.title('Histogram of diff_mat')
    plt.xscale('log')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.savefig('tagates_our_accdiff_hist.png')

bjm = [x for idx, x in tqdm(enumerate(embedding_gen.nb1_api.hash_iterator())) if idx in nb101_train_tagates_sample_indices]
if True:
    accl = [embedding_gen.nb1_api.get_metrics_from_hash(hash_)[1][108][1]['final_validation_accuracy'] for hash_ in tqdm(bjm)]
    kendalltau(accl, tagates_accs)
embedding_gen.nb1_api.get_metrics_from_hash(list(embedding_gen.nb1_api.hash_iterator())[0])
if True:
    tagates_accs = [tagates_train[idx_][-3] for idx_ in range(len(tagates_train))]
    our_accs = [embedding_gen.get_valacc(idx, normalized=False) for idx in nb101_train_tagates_sample_indices]
    diff_mat = np.asarray(tagates_accs) - np.asarray(our_accs)
    norm_our_accs = [embedding_gen.get_valacc(idx, normalized=True) for idx in nb101_train_tagates_sample_indices]
    expr_accs = [embedding_gen.nb1_api.get_metrics_from_hash(hash_)[1][108][0]['final_validation_accuracy'] for hash_ in embedding_gen.nb1_api.hash_iterator()]
    list1 = tagates_accs
    list2 = our_accs
    list3 = norm_our_accs
    plt.figure(figsize=(10, 6))
    flattented_data = np.asarray(list1)
    iqr = np.percentile(flattened_data, 75) - np.percentile(flattened_data, 25)
    bin_width = 2 * iqr * (len(flattened_data) ** (-1/3))
    bin_num = int((flattened_data.max() - flattened_data.min()) / bin_width)
    plt.hist(list1, bins=bin_num, alpha=0.5, label='TAGATES', color='blue')
    flattented_data = np.asarray(list2)
    iqr = np.percentile(flattened_data, 75) - np.percentile(flattened_data, 25)
    bin_width = 2 * iqr * (len(flattened_data) ** (-1/3))
    bin_num = int((flattened_data.max() - flattened_data.min()) / bin_width)
    plt.hist(list2, bins=bin_num, alpha=0.5, label='OurAccs', color='red')
    flattented_data = np.asarray(list3)
    iqr = np.percentile(flattened_data, 75) - np.percentile(flattened_data, 25)
    bin_width = 2 * iqr * (len(flattened_data) ** (-1/3))
    bin_num = int((flattened_data.max() - flattened_data.min()) / bin_width)
    plt.hist(list3, bins=bin_num, alpha=0.3, label='OurNormAccs', color='green')
    plt.title('Histograms of Accuracies of TAGATES and Our Norm/UnNorm Accuracies (NB101)')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig('tagates_our_accs_hist.png')
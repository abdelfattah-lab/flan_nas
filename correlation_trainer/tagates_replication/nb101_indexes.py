import pickle
from tqdm import tqdm
import sys
sys.path.append('/home/ya255/projects/nas_embedding_suite/nas_embedding_suite')
from nb123.nas_bench_101.cell_101 import Cell101
from nasbench import api as NB1API
import numpy as np

def pad_size_6(matrix, ops):
    if len(matrix) < 7:
        new_matrix, new_ops = create_padded_matrix_and_ops(matrix, ops)
        return new_matrix, new_ops
    else:
        return matrix, ops

def create_padded_matrix_and_ops(matrix, ops):
    new_matrix = np.zeros((7, 7), dtype='int8')
    new_ops = []
    n = matrix.shape[0]
    for i in range(7):
        for j in range(7):
            if j < n - 1 and i < n:
                new_matrix[i][j] = matrix[i][j]
            elif j == n - 1 and i < n:
                new_matrix[i][-1] = matrix[i][j]
    for i in range(7):
        if i < n - 1:
            new_ops.append(ops[i])
        elif i < 6:
            new_ops.append('conv3x3-bn-relu')
        else:
            new_ops.append('output')
    return new_matrix, new_ops
    
def transform_nb101_operations(ops):
    # transform_dict = {'input': 0, 'conv1x1-bn-relu': 1, 'conv3x3-bn-relu': 2, 'maxpool3x3': 3, 'output': 4}
    ops = ops[1:-1]
    transform_dict = {'conv1x1-bn-relu': 0, 'conv3x3-bn-relu': 1, 'maxpool3x3': 2}
    ops = [transform_dict[op] for op in ops]
    return ops

# with open('nasbench101_zsall_valid.pkl', "rb") as rf:
#     valid_data = pickle.load(rf)
with open('nasbench101_zsall_train.pkl', "rb") as rf:
    valid_data = pickle.load(rf)

BASE_PATH = '/home/ya255/projects/nas_embedding_suite/nas_embedding_suite/embedding_datasets/'
nb1_api = NB1API.NASBench(BASE_PATH + 'nasbench_only108_caterec.tfrecord')

matrix_mapper = {}
for hash_ in tqdm(nb1_api.hash_iterator()):
    metrics_hashed = nb1_api.get_metrics_from_hash(hash_)
    matrix = metrics_hashed[0]['module_adjacency']
    ops = metrics_hashed[0]['module_operations']
    matrix, ops = pad_size_6(matrix, ops)
    assert(len(ops)==7)
    ops = transform_nb101_operations(ops)
    matrix_mapper[str(matrix.tolist()) + str(ops)] = hash_
    # matrix_mapper[hash_] = {'module_adjacency': matrix, 'module_operations': ops}

hashes_used = []
for vitem in tqdm(valid_data):
    hashes_used.append(
        matrix_mapper[str(vitem[0][0].tolist()) + str(vitem[0][1])]
    )
    # (array([[0, 1, 0, 0, 1, 0, 0],
    #    [0, 0, 1, 0, 0, 0, 0],
    #    [0, 0, 0, 1, 0, 1, 1],
    #    [0, 0, 0, 0, 1, 0, 0],
    #    [0, 0, 0, 0, 0, 1, 0],
    #    [0, 0, 0, 0, 0, 0, 1],
    #    [0, 0, 0, 0, 0, 0, 0]]), [0, 0, 0, 0, 0])
with open("nb101_hash_train.txt", "wb") as fp:   #Pickling
   pickle.dump(hashes_used, fp)

if True:
    import pickle
    with open("nb101_hash_train.txt", "rb") as fp:   #Pickling
        train_set = pickle.load(fp)
    with open("nb101_hash.txt", "rb") as fp:   #Pickling
        val_set = pickle.load(fp)
    print(set(train_set).intersection(set(val_set)))

if True:
    import pickle
    from scipy.stats import kendalltau, spearmanr
    with open('nasbench101_zsall_train.pkl', "rb") as rf:
        train_data = pickle.load(rf)
    with open('nasbench101_zsall_valid.pkl', "rb") as rf:
        valid_data = pickle.load(rf)
    data = train_data
    accuracies = [item[2] for item in data]
    dicts = [item[3] for item in data]
    correlations = {}
    for key in dicts[0].keys():
        values = [d[key] for d in dicts]
        correlation, _ = kendalltau(values, accuracies)
        correlations[key] = correlation
    print("Training ZCP - Output Correlations: ", correlations)
    data = valid_data
    accuracies = [item[2] for item in data]
    dicts = [item[3] for item in data]
    correlations = {}
    for key in dicts[0].keys():
        values = [d[key] for d in dicts]
        correlation, _ = kendalltau(values, accuracies)
        correlations[key] = correlation
    print("Validation ZCP - Output Correlations: ", correlations)

if True:
    import pickle
    import numpy as np
    from scipy.stats import kendalltau, spearmanr
    with open('nasbench101_zsall_train.pkl', "rb") as rf:
        train_data = pickle.load(rf)
    with open('nasbench101_zsall_valid.pkl', "rb") as rf:
        valid_data = pickle.load(rf)
    train_target = [item[2] for item in train_data]
    train_datalist = [{ixn: x for ixn, x in enumerate(list(np.asarray(item[1]).flatten()))} for item in train_data]
    valid_target = [item[2] for item in valid_data]
    valid_datalist = [{ixn: x for ixn, x in enumerate(list(np.asarray(item[1]).flatten()))} for item in valid_data]

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    # Convert list of dictionaries to tensor
    def dict_to_tensor(data_list):
        return torch.tensor([list(item.values()) for item in data_list], dtype=torch.float32)

    train_data_tensor = dict_to_tensor(train_datalist)
    valid_data_tensor = dict_to_tensor(valid_datalist)

    train_target_tensor = torch.tensor(train_target, dtype=torch.float32).view(-1, 1)
    valid_target_tensor = torch.tensor(valid_target, dtype=torch.float32).view(-1, 1)

    # Create DataLoader
    batch_size = 64
    train_dataset = TensorDataset(train_data_tensor, train_target_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = TensorDataset(valid_data_tensor, valid_target_tensor)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # Define a simple neural network
    class SimpleNN(nn.Module):
        def __init__(self, input_size):
            super(SimpleNN, self).__init__()
            self.fc = nn.Sequential(
                nn.Linear(input_size, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
        
        def forward(self, x):
            return self.fc(x)

    model = SimpleNN(len(train_datalist[0]))

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    # Training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        for data, target in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            valid_loss = 0.0
            for data, target in valid_loader:
                outputs = model(data)
                loss = criterion(outputs, target)
                valid_loss += loss.item()
            valid_loss /= len(valid_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {valid_loss:.4f}")

    print("Training complete!")

    model.eval()
    predicted_values = []
    with torch.no_grad():
        for data, _ in valid_loader:
            outputs = model(data)
            predicted_values.extend(outputs.squeeze().tolist())

    # Compute Kendall's tau rank correlation
    correlation, _ = kendalltau(valid_target, predicted_values)
    print(f"Kendall's tau rank correlation: {correlation:.4f}")
    # dicts = [{ixn: x for ixn, x in enumerate(list(np.asarray(item[1]).flatten()))} for item in train_data]
    # correlations = {}
    # for key in dicts[0].keys():
    #     values = [d[key] for d in dicts]
    #     correlation, _ = kendalltau(values, accuracies)
    #     correlations[key] = correlation
    # print("Training ZCP - Output Correlations: ", correlations)
    # print("Max Training ZCP - Output Correlations: ", max([x for x in sorted(correlations.values()) if str(x)!='nan']))
    # accuracies = [item[2] for item in valid_data]
    # dicts = [{ixn: x for ixn, x in enumerate(list(np.asarray(item[1]).flatten()))} for item in valid_data]
    # correlations = {}
    # for key in dicts[0].keys():
    #     values = [d[key] for d in dicts]
    #     correlation, _ = kendalltau(values, accuracies)
    #     correlations[key] = correlation
    # print("Validation ZCP - Output Correlations: ", correlations)
    # print("Max Validation ZCP - Output Correlations: ", max([x for x in sorted(correlations.values()) if str(x)!='nan']))
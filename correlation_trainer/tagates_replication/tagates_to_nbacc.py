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


with open('nasbench101_zsall_valid.pkl', "rb") as rf:
    valid_data = pickle.load(rf)
with open('nasbench101_zsall_train.pkl', "rb") as rf:
    train_data = pickle.load(rf)

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

new_valid_data = valid_data.copy()
new_train_data = train_data.copy()

for idx, vitem in tqdm(enumerate(valid_data)):
    new_valid_data[idx] = list(vitem)
    new_valid_data[idx][2] = nb1_api.get_metrics_from_hash(matrix_mapper[str(vitem[0][0].tolist()) + str(vitem[0][1])])[1][108][1]['final_validation_accuracy']
    new_valid_data[idx] = tuple(new_valid_data[idx])

for idx, vitem in tqdm(enumerate(train_data)):
    new_train_data[idx] = list(vitem)
    new_train_data[idx][2] = nb1_api.get_metrics_from_hash(matrix_mapper[str(vitem[0][0].tolist()) + str(vitem[0][1])])[1][108][1]['final_validation_accuracy']
    new_train_data[idx] = tuple(new_train_data[idx])

with open('nasbench101_zsall_new_valid.pkl', "wb") as wf:
    pickle.dump(new_valid_data, wf)

with open('nasbench101_zsall_new_train.pkl', "wb") as wf:
    pickle.dump(new_train_data, wf)
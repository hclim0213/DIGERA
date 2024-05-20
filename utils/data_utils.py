import numpy as np
import pandas as pd 
import csv

import random
import string
from datetime import date 

import torch
from torch.utils.data import Dataset
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
from .molecules import Molecules

"""
Transformer and LSTM PART 
"""

def data_selection(data_file, meta_file, target_file):
    X = np.load(data_file)
    meta = np.load(meta_file)
    y_data = pd.read_csv(target_file, sep='\t', compression='gzip')
    
    y = y_data.iloc[:, 7:-1].to_numpy(dtype='float32')
    label = list(y_data.iloc[:, -1])
    
    return X, meta, y, label

class CustomDataset(Dataset):
    def __init__(self, x, meta, y):
        self.x = x
        self.meta = meta
        self.y = y
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.x[idx], dtype=torch.int32)
        meta = torch.tensor(self.meta[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)

        return x, meta, y

"""
CIGER PART 
"""

class DataPredict(object):
    def __init__(self, drug_file, gene_file, pert_type, cell_id, pert_idose_type, device):
        self.device = device
        self.fp_type = 'neural'
        self.drug = pd.Series(drug_file[1].values,index=drug_file[0]).to_dict()
        self.input_dim = drug_file.shape[0]
        self.drug_dim = {'atom' : 62, 'bond' : 6}
        self.gene = read_gene(gene_file, 1107, device)
        self.pert_type_feature, self.pert_type_dim, self.use_pert_type = self.pert_type(pert_type)
        self.cell_id_feature, self.cell_id_dim, self.use_cell_id = self.cell_id(cell_id)
        self.pert_idose_feature, self.pert_idose_dim, self.use_pert_idose = self.pert_idose(pert_idose_type)

    def pert_type(self, type):
        pert_temp_list = ['24h', '6h']
        pert_type_dict = dict(zip(pert_temp_list, list(range(len(pert_temp_list)))))
        pert_type_feature = np.zeros(len(pert_temp_list))
        pert_type_feature[pert_type_dict[type]] = 1
        pert_type_feature = np.tile(pert_type_feature, (self.input_dim, 1))
        use_pert_type = True
        pert_type_feature = torch.from_numpy(np.array(pert_type_feature, dtype=np.float32)).to(self.device)
        return pert_type_feature, len(pert_temp_list), use_pert_type # data.pert_type, data.pert_type_dim, use_pert_type 

    def cell_id(self, cell):
        cell_temp_list = [cell]
        cell_id_dict = dict(zip(cell_temp_list, list(range(len(cell_temp_list)))))
        cell_id_feature = np.zeros(len(cell_temp_list))
        cell_id_feature[cell_id_dict[cell]] = 1
        cell_id_feature = np.tile(cell_id_feature, (self.input_dim, 1))
        use_cell_id = True
        cell_id_feature = torch.from_numpy(np.array(cell_id_feature, dtype=np.float32)).to(self.device)
        return cell_id_feature, len(cell_temp_list), use_cell_id # data.pert_type, data.pert_type_dim, use_pert_type 

    def pert_idose(self, type):
        pert_idose_temp_list = ['10um', '5um']
        pert_idose_dict = dict(zip(pert_idose_temp_list, list(range(len(pert_idose_temp_list)))))
        pert_idose_feature = np.zeros(len(pert_idose_temp_list))
        pert_idose_feature[pert_idose_dict[type]] = 1
        pert_idose_feature = np.tile(pert_idose_feature, (self.input_dim, 1))
        use_pert_idose = True
        pert_idose_feature = torch.from_numpy(np.array(pert_idose_feature, dtype=np.float32)).to(self.device)
        return pert_idose_feature, len(pert_idose_temp_list), use_pert_idose # data.pert_type, data.pert_type_dim, use_pert_type 

    def get_feature_from_drug(self):
        output_feature = dict()
        output_feature['drug'] = convert_smile_to_feature(self.drug.keys(), self.device)
        output_feature['mask'] = create_mask_feature(output_feature['drug'], self.device)
        output_feature['pert_type'] = self.pert_type_feature
        output_feature['cell_id'] = self.cell_id_feature
        output_feature['pert_idose'] = self.pert_idose_feature 
        output_feature['gene'] = torch.arange(978).repeat(len(output_feature['cell_id'])).\
                            reshape(len(output_feature['cell_id']), 978).to(self.device)
        return output_feature



def read_drug_number(input_file, num_feature):
    drug = []
    drug_vec = []
    with open(input_file, 'r') as f:
        for line in f:
            #line = line.strip().split(',')
            line = line.strip().split('\t')
            assert len(line) == num_feature + 1, "Wrong format"
            bin_vec = [float(i) for i in line[1:]]
            drug.append(line[0])
            drug_vec.append(bin_vec)
    drug_vec = np.asarray(drug_vec, dtype=np.float32)
    index = []
    for i in range(np.shape(drug_vec)[1]):
        if len(set(drug_vec[:, i])) > 1:
            index.append(i)
    drug_vec = drug_vec[:, index]
    drug = dict(zip(drug, drug_vec))
    return drug, len(index)


def read_drug_string(input_file):
    with open(input_file, 'r') as f:
        drug = dict()
        #for line in csv.reader(f, quotechar='"', delimiter=',', quoting=csv.QUOTE_MINIMAL):
        for line in csv.reader(f, quotechar='"', delimiter='\t', quoting=csv.QUOTE_MINIMAL):
            drug[line[0]] = line[1]
    return drug, {'atom': 62, 'bond': 6}


def read_gene(input_file, num_feature, device):
    with open(input_file, 'r') as f:
        gene = []
        for line in f:
            line = line.strip().split(',')
            #line = line.strip().split('\t')
            assert len(line) == num_feature + 1, "Wrong format"
            gene.append([float(i) for i in line[1:]])
    return torch.from_numpy(np.asarray(gene, dtype=np.float32)).to(device)


def convert_smile_to_feature(smiles, device):
    molecules = Molecules(smiles)
    node_repr = torch.FloatTensor([node.data for node in molecules.get_node_list('atom')]).to(device)
    edge_repr = torch.FloatTensor([node.data for node in molecules.get_node_list('bond')]).to(device)
    return {'molecules': molecules, 'atom': node_repr, 'bond': edge_repr}


def create_mask_feature(data, device):
    batch_idx = data['molecules'].get_neighbor_idx_by_batch('atom')
    molecule_length = [len(idx) for idx in batch_idx]
    mask = torch.zeros(len(batch_idx), max(molecule_length)).to(device)
    for idx, length in enumerate(molecule_length):
        mask[idx][:length] = 1
    return mask


def choose_mean_example(examples):
    num_example = len(examples)
    mean_value = (num_example - 1) / 2
    indexes = np.argsort(examples, axis=0)
    indexes = np.argsort(indexes, axis=0)
    indexes = np.mean(indexes, axis=1)
    distance = (indexes - mean_value)**2
    index = np.argmin(distance)
    return examples[index]


def split_data_by_pert_id_cv(input_file, fold):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    sig_id = []
    label = []
    
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            sig_id.append(line[0])
            label.append(line[1])
    
    sig_id = np.array(sig_id)
    fold_dict = {}
    
    for i, (train_index, test_index) in enumerate(skf.split(sig_id, label)):
        train_pert_id = sig_id[train_index]
        test_pert_id = sig_id[test_index]
        fold_dict[i] = (train_pert_id, test_pert_id)
        
    train_pert_id = fold_dict[fold][0]
    test_pert_id = fold_dict[fold][1]
    return train_pert_id, test_pert_id


def read_data(input_file, drug_train, drug_test, drug):
    feature_train = []
    label_train = []

    feature_test = []
    label_test = []
    data = dict()
    with open(input_file, 'r') as f:
        f.readline()  # skip header
        for line in f:
            line = line.strip().split('\t')
            assert len(line) == 983, "Wrong format"
            if line[0] in drug:
                ft = '\t'.join(line[:1]+line[2:5])
                lb = [float(i) for i in line[5:]]
                if ft in data.keys():
                    data[ft].append(lb)
                else:
                    data[ft] = [lb]

    for ft, lb in data.items():
        ft = ft.split('\t')
        if ft[0] in drug_train:
            feature_train.append(ft)
            if len(lb) == 1:
                label_train.append(lb[0])
            else:
                lb = choose_mean_example(lb)
                label_train.append(lb)
        elif ft[0] in drug_test:
            feature_test.append(ft)
            if len(lb) == 1:
                label_test.append(lb[0])
            else:
                lb = choose_mean_example(lb)
                label_test.append(lb)
        else:
            raise ValueError('Unknown drug')
    label_train = np.array(label_train, dtype=np.float32)
    label_test = np.array(label_test, dtype=np.float32)
    label = np.concatenate([label_train, label_test], axis=0)
    label = np.asarray(label, dtype=np.float32)
    
    pos_threshold = np.quantile(label, 0.95)
    neg_threshold = np.quantile(label, 0.05)
    pos_label_train = np.asarray((label_train > pos_threshold) * 1.0, dtype=np.float32)
    neg_label_train = np.asarray((label_train < neg_threshold) * 1.0, dtype=np.float32)

    pos_label_test = np.asarray((label_test > pos_threshold) * 1.0, dtype=np.float32)
    neg_label_test = np.asarray((label_test < neg_threshold) * 1.0, dtype=np.float32)

    tmp_data = np.asarray(feature_test)
    cell_list = ['A375', 'A549', 'HA1E', 'HCC515', 'HELA', 'HT29', 'MCF7', 'PC3', 'VCAP', 'YAPC']
    cell_idx = []
    for c in cell_list:
        c_idx = tmp_data[:, 2] == c
        cell_idx.append(c_idx)

    return np.asarray(feature_train), np.asarray(feature_test), np.asarray(label_train), \
           np.asarray(label_test), pos_label_train, pos_label_test, \
           neg_label_train, neg_label_test, cell_idx


def transfrom_to_tensor(feature_train, feature_test, label_train, label_test, pos_label_train, 
                        pos_label_test, neg_label_train, neg_label_test, drug, fp_type,
                        device):
    train_drug_feature = []
    test_drug_feature = []
    pert_type_set = sorted(list(set(feature_train[:, 1])))
    cell_id_set = sorted(list(set(feature_train[:, 2])))
    pert_idose_set = sorted(list(set(feature_train[:, 3])))
    use_pert_type = False
    use_cell_id = False
    use_pert_idose = False
    if len(pert_type_set) >= 1:
        pert_type_dict = dict(zip(pert_type_set, list(range(len(pert_type_set)))))
        train_pert_type_feature = []
        test_pert_type_feature = []
        use_pert_type = True
    if len(cell_id_set) >= 1:
        cell_id_dict = dict(zip(cell_id_set, list(range(len(cell_id_set)))))
        train_cell_id_feature = []
        test_cell_id_feature = []
        use_cell_id = True
    if len(pert_idose_set) > 1:
        pert_idose_dict = dict(zip(pert_idose_set, list(range(len(pert_idose_set)))))
        train_pert_idose_feature = []
        test_pert_idose_feature = []
        use_pert_idose = True

    for i, ft in enumerate(feature_train):
        drug_fp = drug[ft[0]]
        train_drug_feature.append(drug_fp)
        if use_pert_type:
            pert_type_feature = np.zeros(len(pert_type_set))
            pert_type_feature[pert_type_dict[ft[1]]] = 1
            train_pert_type_feature.append(np.array(pert_type_feature, dtype=np.float32))
        if use_cell_id:
            cell_id_feature = np.zeros(len(cell_id_set))
            cell_id_feature[cell_id_dict[ft[2]]] = 1
            train_cell_id_feature.append(np.array(cell_id_feature, dtype=np.float32))
        if use_pert_idose:
            pert_idose_feature = np.zeros(len(pert_idose_set))
            pert_idose_feature[pert_idose_dict[ft[3]]] = 1
            train_pert_idose_feature.append(np.array(pert_idose_feature, dtype=np.float32))

    for i, ft in enumerate(feature_test):
        drug_fp = drug[ft[0]]
        test_drug_feature.append(drug_fp)
        if use_pert_type:
            pert_type_feature = np.zeros(len(pert_type_set))
            pert_type_feature[pert_type_dict[ft[1]]] = 1
            test_pert_type_feature.append(np.array(pert_type_feature, dtype=np.float32))
        if use_cell_id:
            cell_id_feature = np.zeros(len(cell_id_set))
            cell_id_feature[cell_id_dict[ft[2]]] = 1
            test_cell_id_feature.append(np.array(cell_id_feature, dtype=np.float32))
        if use_pert_idose:
            pert_idose_feature = np.zeros(len(pert_idose_set))
            pert_idose_feature[pert_idose_dict[ft[3]]] = 1
            test_pert_idose_feature.append(np.array(pert_idose_feature, dtype=np.float32))

    train_feature = dict()
    test_feature = dict()
    train_label = dict()
    test_label = dict()
    if fp_type == 'ecfp':
        train_feature['drug'] = torch.from_numpy(np.asarray(train_drug_feature, dtype=np.float32)).to(device)
        test_feature['drug'] = torch.from_numpy(np.asarray(test_drug_feature, dtype=np.float32)).to(device)
    elif fp_type == 'all':
        train_feature['drug'] = torch.from_numpy(np.asarray(train_drug_feature, dtype=np.float32)).to(device)
        test_feature['drug'] = torch.from_numpy(np.asarray(test_drug_feature, dtype=np.float32)).to(device)
    elif fp_type == 'neural':
        train_feature['drug'] = np.asarray(train_drug_feature)
        test_feature['drug'] = np.asarray(test_drug_feature)
    if use_pert_type:
        train_feature['pert_type'] = torch.from_numpy(np.asarray(train_pert_type_feature, dtype=np.float32)).to(device)
        test_feature['pert_type'] = torch.from_numpy(np.asarray(test_pert_type_feature, dtype=np.float32)).to(device)
    if use_cell_id:
        train_feature['cell_id'] = torch.from_numpy(np.asarray(train_cell_id_feature, dtype=np.float32)).to(device)
        test_feature['cell_id'] = torch.from_numpy(np.asarray(test_cell_id_feature, dtype=np.float32)).to(device)
    if use_pert_idose:
        train_feature['pert_idose'] = torch.from_numpy(np.asarray(train_pert_idose_feature, dtype=np.float32)).to(device)
        test_feature['pert_idose'] = torch.from_numpy(np.asarray(test_pert_idose_feature, dtype=np.float32)).to(device)
    
    train_label['real'] = torch.from_numpy(label_train).to(device)
    test_label['real'] = torch.from_numpy(label_test).to(device)
    train_label['binary'] = torch.from_numpy(pos_label_train).to(device)
    test_label['binary'] = torch.from_numpy(pos_label_test).to(device)
    train_label['binary_reverse'] = torch.from_numpy(neg_label_train).to(device)
    test_label['binary_reverse'] = torch.from_numpy(neg_label_test).to(device)
    return train_feature, test_feature, train_label, test_label, use_pert_type, use_cell_id, \
           use_pert_idose, len(pert_type_set), len(cell_id_set), len(pert_idose_set)

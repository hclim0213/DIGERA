from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from model import Transformer, LSTMModel, Graph_Transformer, LinearRegression_3d, CustomRandomForestRegressor
from utils import DataReader, data_selection, CustomDataset
from utils import custom_make_rank, custom_score, custom_top_k_score

import pickle
import joblib
import argparse
import warnings
warnings.filterwarnings(action='ignore')

# setting configurations
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config={'transformer_d_model': 64,
        'transformer_d_hid' : 1024,
        'transformer_head' : 1, 
        'lstm_d_model': 512, 
        'lstm_d_hid': 128,
        }

train_index_dict = {'MCF7':[], 'HA1E':[], 'A375':[], 
             'HEPG2':[], 'A549':[], 'HT29':[], 'VCAP':[], 
             'PC3':[], 'three_cells':[], 'eight_cells':[]}

test_index_dict = {'MCF7':[], 'HA1E':[], 'A375':[], 
             'HEPG2':[], 'A549':[], 'HT29':[], 'VCAP':[], 
             'PC3':[], 'three_cells':[], 'eight_cells':[]}

data_v1_X = pd.read_csv('data/Bayesian_phase2_L5_All_num_features.tsv.gz', sep='\t', compression='gzip')

list_selected_labels_arr = {
        'MCF7': (98, 20),
        'HA1E': (10, 45),
        'A375': (81, 49), 
        'HEPG2': (15, 48), 
        'A549': (14, 35),
        'HT29': (28, 41),
        'VCAP': (26, 19),
        'PC3': (17, 31), 
        'three_cells': (98, 20, 26, 19, 17, 31),
        'eight_cells': (49, 81, 14, 35, 10, 45, 15, 48, 28, 41, 98, 20, 17, 31, 19, 26), 
    }

for cell, selected_labels_arr in list_selected_labels_arr.items():
    np_X = data_v1_X[data_v1_X['KNN_labels'].isin(selected_labels_arr)].drop(columns='KNN_labels').iloc[:, 6:].to_numpy()
    selected_labels = data_v1_X[data_v1_X['KNN_labels'].isin(selected_labels_arr)]['KNN_labels'].to_numpy()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    for i, (train_index, test_index) in enumerate(skf.split(np_X, selected_labels)):
        train_index_dict[cell].append(train_index)
        test_index_dict[cell].append(test_index)

def RandomForest_predict(cell, fold):

    rfr_estimator = 2500
    rfr_depth = 20
    
    RF_X_data = pd.read_csv('data/Bayesian_phase2_L5_All_num_features.tsv.gz', sep='\t', compression='gzip')
    RF_y_data = pd.read_csv('data/Bayesian_all_phase2_L5_rank.tsv.gz', sep='\t', compression='gzip')

    selected_labels_arr = list_selected_labels_arr[cell]
    selected_labels = RF_X_data[RF_X_data['KNN_labels'].isin(selected_labels_arr)]['KNN_labels'].to_numpy()
    
    np_X = RF_X_data[RF_X_data['KNN_labels'].isin(selected_labels_arr)].drop(columns='KNN_labels').iloc[:, 6:].to_numpy()
    np_y = RF_y_data[RF_y_data['KNN_labels'].isin(selected_labels_arr)].drop(columns='KNN_labels').iloc[:, 7:].to_numpy()

    X_train, X_test = np_X[train_index_dict[cell][fold], :], np_X[test_index_dict[cell][fold], :]
    train_y, test_y = np_y[train_index_dict[cell][fold], :], np_y[test_index_dict[cell][fold], :]

    file_path = f"saved_model/RFR/RandomForest_cv_{fold}_{cell}.joblib.gz"
    model_rf = joblib.load(file_path)
    model_rf.set_params(verbose=0)
    
    train_pred_rank = model_rf.predict(X_train)
    test_pred_rank = model_rf.predict(X_test)

    return [train_y, train_pred_rank], [test_y, test_pred_rank]
                                                        
def Transformer_pred(X, y, meta, cell, loss_type, fold, batch_size):
    input_dim = X.shape[1:]
    output_dim = y.shape[1]
    meta_dim = meta.shape[1]

    dataset = CustomDataset(X, meta, y)

    train_data = torch.utils.data.Subset(dataset, train_index_dict[cell][fold])
    test_data = torch.utils.data.Subset(dataset, test_index_dict[cell][fold])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)
    
    model = Transformer(input_dim=input_dim, d_model=config['transformer_d_model'], output_dim=output_dim, meta_dim = meta_dim,
                        loss_type=loss_type, nhead=1, d_hid=config['transformer_d_hid'], nlayers=1, dropout=0.01, device=device).to(device)

    checkpoint = torch.load(f'saved_model/SMILES_Transformer/Transformer_{loss_type}_rank_{cell}_{fold}.ckpt')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    iter_n = 0 
    with torch.no_grad():
        for batch in train_loader:
            x = batch[0].to(device)
            cond = batch[1].to(device)
            y = batch[2].detach().cpu().numpy()
            pred_rank = model.predict(x, cond) 

            if iter_n == 0:
                pred_rank_np = np.array(pred_rank)
                true_y = y
            else:
                pred_rank_np = np.append(pred_rank_np, np.array(pred_rank), axis=0)
                true_y = np.append(true_y, y, axis=0)
            iter_n+=1

    train_pred_rank = pred_rank_np
    train_y = true_y

    iter_n = 0
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            x = batch[0].to(device)
            cond = batch[1].to(device)
            y = batch[2].detach().cpu().numpy()
            pred_rank = model.predict(x, cond)

            if iter_n == 0:
                pred_rank_np = np.array(pred_rank)
                true_y = y
            else:
                pred_rank_np = np.append(pred_rank_np, np.array(pred_rank), axis=0)
                true_y = np.append(true_y, y, axis=0)
            iter_n+=1

    test_pred_rank = pred_rank_np
    test_y = true_y
    
    return [train_y, train_pred_rank], [test_y, test_pred_rank]

def LSTM_pred(X, y, meta, cell, loss_type, fold, batch_size):
    input_dim = X.shape[1:]
    output_dim = y.shape[1]
    ntoken = input_dim[1]
    meta_dim = meta.shape[1]

    dataset = CustomDataset(X, meta, y)

    train_data = torch.utils.data.Subset(dataset, train_index_dict[cell][fold])
    test_data = torch.utils.data.Subset(dataset, test_index_dict[cell][fold])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)
    
    model = LSTMModel(input_dim=input_dim, d_model=config['lstm_d_model'], output_dim=output_dim, meta_dim = meta_dim, 
                      loss_type=loss_type, d_hid=config['lstm_d_hid'], nlayers=1, dropout=0.01, device=device).to(device)

    checkpoint = torch.load(f'saved_model/SMILES_LSTM/LSTM_{loss_type}_rank_{cell}_{fold}.ckpt')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    iter_n = 0 
    with torch.no_grad():
        for batch in train_loader:
            x = batch[0].to(device)
            cond = batch[1].to(device)
            y = batch[2].detach().cpu().numpy()
            
            hidden = model.init_hidden(ntoken) # batch_size but size error 
            pred_rank = model.predict(x, cond, hidden)

            if iter_n == 0:
                pred_rank_np = np.array(pred_rank)
                true_y = y
            else:
                pred_rank_np = np.append(pred_rank_np, np.array(pred_rank), axis=0)
                true_y = np.append(true_y, y, axis=0)
            iter_n+=1

    train_pred_rank = pred_rank_np
    train_y = true_y

    iter_n = 0

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            x = batch[0].to(device)
            cond = batch[1].to(device)
            y = batch[2].detach().cpu().numpy()
            
            hidden = model.init_hidden(ntoken) # batch_size but size error 
            pred_rank = model.predict(x, cond, hidden)

            if iter_n == 0:
                pred_rank_np = np.array(pred_rank)
                true_y = y
            else:
                pred_rank_np = np.append(pred_rank_np, np.array(pred_rank), axis=0)
                true_y = np.append(true_y, y, axis=0)
            iter_n+=1

    test_pred_rank = pred_rank_np
    test_y = true_y

    return [train_y, train_pred_rank], [test_y, test_pred_rank]

def GTF_pred(drug_file, drug_id_file, gene_file, data_file, 
            cell, loss_type, fold, batch_size, n_layers, n_heads):
    fp_type = 'neural'
    label_type = 'real'
    num_gene = 978

    data = DataReader(drug_file, drug_id_file, gene_file, data_file, fp_type, device, fold)
    intitializer = torch.nn.init.xavier_uniform_

    graph_transformer = Graph_Transformer(drug_input_dim= data.drug_dim, gene_embed=data.gene, gene_input_dim=data.gene.size()[1],
                                        n_layers=n_layers, n_heads=n_heads, 
                                        encode_dim=512, fp_type=fp_type, loss_type=loss_type, label_type=label_type, device=device,
                                        initializer=intitializer, pert_type_input_dim=data.pert_type_dim, cell_id_input_dim=data.cell_id_dim,
                                        pert_idose_input_dim=data.pert_idose_dim, use_pert_type=data.use_pert_type,
                                        use_cell_id=data.use_cell_id, use_pert_idose=data.use_pert_idose)

    checkpoint = torch.load(f'saved_model/Graph_Transformer/Transformer_{loss_type}_real_{cell}_{fold}.ckpt')
    graph_transformer.load_state_dict(checkpoint['model_state_dict'])
    graph_transformer.to(device)
    graph_transformer.eval()
    
    label_binary_np = np.empty([0, num_gene])
    label_real_np = np.empty([0, num_gene])
    predict_rank_np = np.empty([0, num_gene])

    with torch.no_grad():
        for i, batch in enumerate(data.get_batch_data(dataset='train', batch_size=batch_size, shuffle=False)):
            ft, lb = batch
            drug = ft['drug']
            if data.use_pert_type:
                pert_type = ft['pert_type']
            else:
                pert_type = None
            if data.use_cell_id:
                cell_id = ft['cell_id']
            else:
                cell_id = None
            if data.use_pert_idose:
                pert_idose = ft['pert_idose']
            else:
                pert_idose = None
            gene = ft['gene']
            predict_rank = graph_transformer.predict(drug, gene, pert_type, cell_id, pert_idose)
            if label_type == 'binary' or label_type == 'real':
                label = lb['binary']
            elif label_type == 'binary_reverse' or label_type == 'real_reverse':
                label = lb['binary_reverse']
            else:
                raise ValueError('Unknown label type : %s' % label_type)
    
            if label_type == 'binary' or label_type == 'real':
                label_binary = lb['binary']
                label_real = lb['real']
            elif label_type == 'binary_reverse' or label_type == 'real_reverse':
                label_binary = lb['binary_reverse']
                label_real = -lb['real']
            else:
                raise ValueError('Unknown label type: %s' % label_type)
                
            label_binary_np = np.concatenate((label_binary_np, label_binary.cpu().numpy()), axis=0)
            label_real_np = np.concatenate((label_real_np, label_real.cpu().numpy()), axis=0)
            predict_rank_np = np.concatenate((predict_rank_np, predict_rank), axis=0)

    label_real_np = np.where(label_real_np < 0, 0, label_real_np)
    predict_rank_np = np.where(predict_rank_np < 0, 0, predict_rank_np)

    train_pred_rank = predict_rank_np
    train_y = label_real_np 

    label_binary_np = np.empty([0, num_gene])
    label_real_np = np.empty([0, num_gene])
    predict_rank_np = np.empty([0, num_gene])

    with torch.no_grad():
        for i, batch in enumerate(data.get_batch_data(dataset='test', batch_size=batch_size, shuffle=False)):
            ft, lb = batch
            drug = ft['drug']
            if data.use_pert_type:
                pert_type = ft['pert_type']
            else:
                pert_type = None
            if data.use_cell_id:
                cell_id = ft['cell_id']
            else:
                cell_id = None
            if data.use_pert_idose:
                pert_idose = ft['pert_idose']
            else:
                pert_idose = None
            gene = ft['gene']
            predict_rank = graph_transformer.predict(drug, gene, pert_type, cell_id, pert_idose)
            if label_type == 'binary' or label_type == 'real':
                label = lb['binary']
            elif label_type == 'binary_reverse' or label_type == 'real_reverse':
                label = lb['binary_reverse']
            else:
                raise ValueError('Unknown label type : %s' % label_type)
    
            if label_type == 'binary' or label_type == 'real':
                label_binary = lb['binary']
                label_real = lb['real']
            elif label_type == 'binary_reverse' or label_type == 'real_reverse':
                label_binary = lb['binary_reverse']
                label_real = -lb['real']
            else:
                raise ValueError('Unknown label type: %s' % label_type)
                
            label_binary_np = np.concatenate((label_binary_np, label_binary.cpu().numpy()), axis=0)
            label_real_np = np.concatenate((label_real_np, label_real.cpu().numpy()), axis=0)
            predict_rank_np = np.concatenate((predict_rank_np, predict_rank), axis=0)

    label_real_np = np.where(label_real_np < 0, 0, label_real_np)
    predict_rank_np = np.where(predict_rank_np < 0, 0, predict_rank_np)

    test_pred_rank = predict_rank_np
    test_y = label_real_np 

    return [train_y, train_pred_rank], [test_y, test_pred_rank]

def train(data_file, meta_file, target_file, gtf_drug_file, gtf_drug_id_file, gtf_gene_file, gtf_data_file, 
        cell, fold, batch_size, penalty, model_name):
    X, meta, y, label = data_selection(data_file, meta_file, target_file)
    real_train_y = y[train_index_dict[cell][fold]]
    real_test_y = y[test_index_dict[cell][fold]]
    
    rpred_train, rpred_test = RandomForest_predict(cell, fold)
    tf_train, tf_test = Transformer_pred(X, y, meta, cell, 'list_wise_rankcosine', fold, batch_size)
    lstm_train, lstm_test = LSTM_pred(X, y, meta, cell, 'list_wise_rankcosine', fold, batch_size)
    gtf_train, gtf_test = GTF_pred(gtf_drug_file, gtf_drug_id_file, gtf_gene_file, gtf_data_file, cell, 
                                   'point_wise_mse', fold, batch_size, 2, 1)
    
    print(f"CV\t{fold}\tCell\t{cell}\tTrain Shape\tRF : {rpred_train[0].shape}/ SMILES_Transformer : {tf_train[1].shape}/ SMILES_LSTM : {lstm_train[1].shape}/ Graph_Transformer : {gtf_train[1].shape}") 
    print(f"CV\t{fold}\tCell\t{cell}\tTest Shape\tRF : {rpred_test[0].shape}/ SMILES_Transformer : {tf_test[1].shape}/ SMILES_LSTM : {lstm_test[1].shape}/ Graph_Transformer : {gtf_test[1].shape}") 
    
    train_rank_stack = np.stack((rpred_train[0], tf_train[1], lstm_train[1], gtf_train[1]), axis=2)
    test_rank_stack = np.stack((rpred_test[0], tf_test[1], lstm_test[1], gtf_test[1]), axis=2)

    c = -1
    l = -1
    if penalty == 'l1':
        c = 0.3
        l = 0.01
    elif penalty == 'l2':
        c = 0.3
        l = 0.01
    elif penalty == 'l3':
        c = 0.3
        l = 0.01
        
    # multioutput Ridge
    meta_model = LinearRegression_3d(C=c, lr=l, penalty=penalty)
    meta_model.fit(train_rank_stack, real_train_y)
    
    with torch.no_grad():
        pred_train = meta_model.predict(train_rank_stack)
        pred_test = meta_model.predict(test_rank_stack)
    
    pred_train_rank = custom_make_rank(pred_train).squeeze()
    pred_test_rank = custom_make_rank(pred_test).squeeze()
    
    f1score = custom_score(real_train_y, pred_train_rank)
    print(f"CV\t{fold}\tCell\t{cell}\tTrain\tF1score\t{f1score}")
    train_topk = custom_top_k_score(real_train_y, pred_train_rank)
    print(f"CV\t{fold}\tCell\t{cell}\tTrain\tCustom_topk_v2_Up_Precision\t{np.char.mod('%f', np.reshape(train_topk, -1)[0:5])}")
    print(f"CV\t{fold}\tCell\t{cell}\tTrain\tCustom_topk_v2_Up_Recall\t{np.char.mod('%f', np.reshape(train_topk, -1)[5:10])}")
    print(f"CV\t{fold}\tCell\t{cell}\tTrain\tCustom_topk_v2_Up_F1score\t{np.char.mod('%f', np.reshape(train_topk, -1)[10:15])}")
    print(f"CV\t{fold}\tCell\t{cell}\tTrain\tCustom_topk_v2_Down_Precision\t{np.char.mod('%f', np.reshape(train_topk, -1)[15:20])}")
    print(f"CV\t{fold}\tCell\t{cell}\tTrain\tCustom_topk_v2_Down_Recall\t{np.char.mod('%f', np.reshape(train_topk, -1)[20:25])}")
    print(f"CV\t{fold}\tCell\t{cell}\tTrain\tCustom_topk_v2_Down_F1score\t{np.char.mod('%f', np.reshape(train_topk, -1)[25:30])}")
    print(f"CV\t{fold}\tCell\t{cell}\tTrain\tCustom_topk_v2_Up_Down_Precision\t{np.char.mod('%f', np.reshape(train_topk, -1)[30:35])}")
    print(f"CV\t{fold}\tCell\t{cell}\tTrain\tCustom_topk_v2_Up_Down_Recall\t{np.char.mod('%f', np.reshape(train_topk, -1)[35:40])}")
    print(f"CV\t{fold}\tCell\t{cell}\tTrain\tCustom_topk_v2_Up_Down_F1score\t{np.char.mod('%f', np.reshape(train_topk, -1)[40:45])}")
    
    f1score = custom_score(real_test_y, pred_test_rank)
    print(f"CV\t{fold}\tCell\t{cell}\tTest\tF1score\t{f1score}")
    test_topk = custom_top_k_score(real_test_y, pred_test_rank)
    print(f"CV\t{fold}\tCell\t{cell}\tTest\tCustom_topk_v2_Up_Precision\t{np.char.mod('%f', np.reshape(test_topk, -1)[0:5])}")
    print(f"CV\t{fold}\tCell\t{cell}\tTest\tCustom_topk_v2_Up_Recall\t{np.char.mod('%f', np.reshape(test_topk, -1)[5:10])}")
    print(f"CV\t{fold}\tCell\t{cell}\tTest\tCustom_topk_v2_Up_F1score\t{np.char.mod('%f', np.reshape(test_topk, -1)[10:15])}")
    print(f"CV\t{fold}\tCell\t{cell}\tTest\tCustom_topk_v2_Down_Precision\t{np.char.mod('%f', np.reshape(test_topk, -1)[15:20])}")
    print(f"CV\t{fold}\tCell\t{cell}\tTest\tCustom_topk_v2_Down_Recall\t{np.char.mod('%f', np.reshape(test_topk, -1)[20:25])}")
    print(f"CV\t{fold}\tCell\t{cell}\tTest\tCustom_topk_v2_Down_F1score\t{np.char.mod('%f', np.reshape(test_topk, -1)[25:30])}")
    print(f"CV\t{fold}\tCell\t{cell}\tTest\tCustom_topk_v2_Up_Down_Precision\t{np.char.mod('%f', np.reshape(test_topk, -1)[30:35])}")
    print(f"CV\t{fold}\tCell\t{cell}\tTest\tCustom_topk_v2_Up_Down_Recall\t{np.char.mod('%f', np.reshape(test_topk, -1)[35:40])}")
    print(f"CV\t{fold}\tCell\t{cell}\tTest\tCustom_topk_v2_Up_Down_F1score\t{np.char.mod('%f', np.reshape(test_topk, -1)[40:45])}")

    joblib.dump(meta_model, f"saved_model/Ensemble/{model_name}_cv_{fold}_{cell}.pkl")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Lasso, Ridge, and Elastic Regression')
    parser.add_argument('--cell_type', help='MCF7/HA1E/A375...etc')
    parser.add_argument('--model_name', help='Lasso/Ridge/Elastic')
    parser.add_argument('--batch_size', help='Data Number/Train')
    
    parser.add_argument('--data_file', help='X data path')
    parser.add_argument('--meta_file', help='meta data path, dosage and time condition')
    parser.add_argument('--target_file', help='y data path')

    parser.add_argument('--gtf_drug_file', help='Graph Transformer drug features ecfp or smiles')
    parser.add_argument('--gtf_drug_id_file', help='Graph Transformer drug id file')
    parser.add_argument('--gtf_gene_file', help='Graph Transformer gene file path')
    parser.add_argument('--gtf_data_file', help='Graph Transformer target data file path')
    
    args = parser.parse_args()
    
    cell_type = args.cell_type
    model_name = args.model_name
    batch_size = int(args.batch_size)
    
    data_file = args.data_file
    meta_file = args.meta_file
    target_file = args.target_file 

    gtf_drug_file = args.gtf_drug_file
    gtf_drug_id_file = args.gtf_drug_id_file
    gtf_gene_file = args.gtf_gene_file
    gtf_data_file = args.gtf_data_file

    if model_name == 'Lasso':
        penalty = 'l2'
    elif model_name == 'Ridge':
        penalty = 'l1'
    elif model_name == 'Elastic':
        penalty = 'l3'

    for fold in range(5):
        train(data_file, meta_file, target_file, 
              gtf_drug_file, gtf_drug_id_file, gtf_gene_file, gtf_data_file, 
              cell_type, fold, batch_size, penalty, model_name)
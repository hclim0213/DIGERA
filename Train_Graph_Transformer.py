import torch
import numpy as np
import random
import os

from datetime import datetime
import argparse
import math
from tqdm import tqdm
from model import Graph_Transformer
from utils import DataReader, custom_top_k_score, custom_score

seed = 42
np.random.seed(seed=seed)
random.seed(a=seed)
torch.manual_seed(seed)

if __name__=="__main__":
    start_time = datetime.now()

    parser = argparse.ArgumentParser(description='Graph Transformer Training')
    parser.add_argument('--drug_file', help='drug feature file (ECFP or SMILES)')
    parser.add_argument('--drug_id_file', help='drug id file')
    parser.add_argument('--gene_file', help='gene feature file')
    parser.add_argument('--data_file', help='chemical signature file')
    parser.add_argument('--fp_type', help='ECFP or Neural FP')
    parser.add_argument('--label_type', help='real/real reverse/binary/binary reverse')
    parser.add_argument('--loss_type', help='point_wise_mse/list_wise_rankcosine')
    parser.add_argument('--batch_size', help='number of training example per update')
    parser.add_argument('--max_epoch', help='total number of training iterations')
    parser.add_argument('--lr', help='learning rate')
    parser.add_argument('--fold', help='id for testing set in cross-validation setting')
    parser.add_argument('--warm_start', help='training from pre-trained model')
    parser.add_argument('--inference', help='inference from pre-trained model')
    parser.add_argument('--cell_type', help='A549, etc')

    args = parser.parse_args()

    drug_file = args.drug_file
    drug_id_file = args.drug_id_file
    gene_file = args.gene_file
    data_file = args.data_file
    fp_type = args.fp_type
    label_type = args.label_type
    loss_type = args.loss_type
    batch_size = int(args.batch_size)
    max_epoch = int(args.max_epoch)
    lr = float(args.lr)
    fold = int(args.fold)
    warm_start = True if args.warm_start == 'True' else False
    inference = True if args.inference == 'True' else False
    cell_type = args.cell_type

    model_name = 'Graph_Transformer'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    intitializer = torch.nn.init.xavier_uniform_
    num_gene = 978
    data = DataReader(drug_file, drug_id_file, gene_file, data_file, fp_type, device, fold)
    print('#Train: %d' % len(data.train_feature['drug']))
    print('#Test: %d' % len(data.test_feature['drug']))

    if inference:
        model = Graph_Transformer(drug_input_dim=data.drug_dim, gene_embed=data.gene, gene_input_dim=data.gene.size()[1],
                    encode_dim=512, fp_type=fp_type, loss_type=loss_type, label_type=label_type, device=device,
                    initializer=intitializer, pert_type_input_dim=data.pert_type_dim, cell_id_input_dim=data.cell_id_dim,
                    pert_idose_input_dim=data.pert_idose_dim, use_pert_type=data.use_pert_type,
                    use_cell_id=data.use_cell_id, use_pert_idose=data.use_pert_idose)
        checkpoint = torch.load('saved_model/graph_transformer/%s_%s_%d.ckpt' % (model_name + '_' + loss_type + '_' + label_type, \
                                                                    cell_type, fold),
                                map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        epoch_loss = 0
        label_binary_np = np.empty([0, num_gene])
        label_real_np = np.empty([0, num_gene])
        predict_np = np.empty([0, num_gene])
        with torch.no_grad():
            for i, batch in enumerate(tqdm(data.get_batch_data(dataset='test', batch_size=batch_size, shuffle=False))):
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
                predict = model(drug, gene, pert_type, cell_id, pert_idose)
                if label_type == 'binary' or label_type == 'real':
                    label = lb['binary']
                elif label_type == 'binary_reverse' or label_type == 'real_reverse':
                    label = lb['binary_reverse']
                else:
                    raise ValueError('Unknown label type: %s' % label_type)

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
                predict_np = np.concatenate((predict_np, predict.cpu().numpy()), axis=0)
                
            label_real_np = np.where(label_real_np < 0, 0, label_real_np)
            predict_np = np.where(predict_np < 0, 0, predict_np)

    else:
        model = Graph_Transformer(drug_input_dim=data.drug_dim, gene_embed=data.gene, gene_input_dim=data.gene.size()[1],
                    encode_dim=512, fp_type=fp_type, loss_type=loss_type, label_type=label_type, device=device,
                    initializer=intitializer, pert_type_input_dim=data.pert_type_dim, cell_id_input_dim=data.cell_id_dim,
                    pert_idose_input_dim=data.pert_idose_dim, use_pert_type=data.use_pert_type,
                    use_cell_id=data.use_cell_id, use_pert_idose=data.use_pert_idose)

        if warm_start:     
            checkpoint = torch.load('saved_model/Graph_Transformer/%s_%s_%d.ckpt' % (model_name + '_' + loss_type + '_' + label_type, \
                                                                    cell_type, fold), map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        best_score = float("-inf")
        
        score_list_train = {'f1score':[], 'up_precision':[],      'up_recall':[],      'up_f1score':[], \
                            'down_precision':[],    'down_recall':[],    'down_f1score':[], \
                            'up_down_precision':[], 'up_down_recall':[], 'up_down_f1score':[], 'custom_top_k1':[]\
                         }
    
        score_list_test = {'f1score':[], 'up_precision':[],      'up_recall':[],      'up_f1score':[], \
                            'down_precision':[],    'down_recall':[],    'down_f1score':[], \
                            'up_down_precision':[], 'up_down_recall':[], 'up_down_f1score':[], 'custom_top_k1':[]\
                         }

        num_batch_train = math.ceil(len(data.train_feature['drug']) / batch_size)
        num_batch_test = math.ceil(len(data.test_feature['drug']) / batch_size)
        
        """
        
        Train
        
        """
        for epoch in range(max_epoch):
            model.train()
            epoch_loss = 0
            for i, batch in enumerate(tqdm(data.get_batch_data(dataset='train', batch_size=batch_size, shuffle=True),
                                        total=num_batch_train)):
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
                if label_type == 'binary':
                    label = lb['binary']
                elif label_type == 'binary_reverse':
                    label = lb['binary_reverse']
                elif label_type == 'real':
                    label = lb['real']
                elif label_type == 'real_reverse':
                    label = -lb['real']
                else:
                    raise ValueError('Unknown label type: %s' % label_type)
                optimizer.zero_grad()
                predict = model(drug, gene, pert_type, cell_id, pert_idose)
                loss = model.loss(label, predict)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            model.eval()

            label_binary_np = np.empty([0, num_gene])
            label_real_np = np.empty([0, num_gene])
            predict_rank_np = np.empty([0, num_gene])

            with torch.no_grad():
                for i, batch in enumerate(tqdm(data.get_batch_data(dataset='train', batch_size=batch_size, shuffle=False),
                                            total=num_batch_train)):
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
                    predict_rank, predict_median = model.predict(drug, gene, pert_type, cell_id, pert_idose)
                    if label_type == 'binary' or label_type == 'real':
                        label = lb['binary']
                    elif label_type == 'binary_reverse' or label_type == 'real_reverse':
                        label = lb['binary_reverse']
                    else:
                        raise ValueError('Unknown label type: %s' % label_type)

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
                            
                f1score = custom_score(label_real_np, predict_rank_np)
                score_list_train['f1score'].append(f1score)

                topk = custom_top_k_score(label_real_np, predict_rank_np)
                score_list_train['up_precision'].append(np.reshape(topk, -1)[0:5])
                score_list_train['up_recall'].append(np.reshape(topk, -1)[5:10])
                score_list_train['up_f1score'].append(np.reshape(topk, -1)[10:15])
                score_list_train['down_precision'].append(np.reshape(topk, -1)[15:20])
                score_list_train['down_recall'].append(np.reshape(topk, -1)[20:25])
                score_list_train['down_f1score'].append(np.reshape(topk, -1)[25:30])
                score_list_train['up_down_precision'].append(np.reshape(topk, -1)[30:35])
                score_list_train['up_down_recall'].append(np.reshape(topk, -1)[35:40])
                score_list_train['up_down_f1score'].append(np.reshape(topk, -1)[40:45])
                                
                print(f"CV\t{fold}\tEpoch\tTrain\t{epoch+1}\tF1score\t{f1score}")
                print(f"CV\t{fold}\tEpoch\tTrain\t{epoch+1}\tUp Precision\t{np.reshape(topk, -1)[0:5]}")
                print(f"CV\t{fold}\tEpoch\tTrain\t{epoch+1}\tUp Recall\t{np.reshape(topk, -1)[5:10]}")
                print(f"CV\t{fold}\tEpoch\tTrain\t{epoch+1}\tUp F1score\t{np.reshape(topk, -1)[10:15]}")
                print(f"CV\t{fold}\tEpoch\tTrain\t{epoch+1}\tDown Precision\t{np.reshape(topk, -1)[15:20]}")
                print(f"CV\t{fold}\tEpoch\tTrain\t{epoch+1}\tDown Recall\t{np.reshape(topk, -1)[20:25]}")
                print(f"CV\t{fold}\tEpoch\tTrain\t{epoch+1}\tDown F1score\t{np.reshape(topk, -1)[25:30]}")
                print(f"CV\t{fold}\tEpoch\tTrain\t{epoch+1}\tUp_Down Precision\t{np.reshape(topk, -1)[30:35]}")
                print(f"CV\t{fold}\tEpoch\tTrain\t{epoch+1}\tUp_Down Recall\t{np.reshape(topk, -1)[35:40]}")
                print(f"CV\t{fold}\tEpoch\tTrain\t{epoch+1}\tUp_Down F1score\t{np.reshape(topk, -1)[40:45]}")

            label_binary_np = np.empty([0, num_gene])
            label_real_np = np.empty([0, num_gene])
            predict_rank_np = np.empty([0, num_gene])

            with torch.no_grad():
                for i, batch in enumerate(tqdm(data.get_batch_data(dataset='test', batch_size=batch_size, shuffle=False),
                                            total=num_batch_test)):
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
                    predict_rank, predict_median = model.predict(drug, gene, pert_type, cell_id, pert_idose)
                    if label_type == 'binary' or label_type == 'real':
                        label = lb['binary']
                    elif label_type == 'binary_reverse' or label_type == 'real_reverse':
                        label = lb['binary_reverse']
                    else:
                        raise ValueError('Unknown label type: %s' % label_type)

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
                
                f1score = custom_score(label_real_np, predict_rank_np)
                score_list_test['f1score'].append(f1score)

                topk = custom_top_k_score(label_real_np, predict_rank_np)
                score_list_test['up_precision'].append(np.reshape(topk, -1)[0:5])
                score_list_test['up_recall'].append(np.reshape(topk, -1)[5:10])
                score_list_test['up_f1score'].append(np.reshape(topk, -1)[10:15])
                score_list_test['down_precision'].append(np.reshape(topk, -1)[15:20])
                score_list_test['down_recall'].append(np.reshape(topk, -1)[20:25])
                score_list_test['down_f1score'].append(np.reshape(topk, -1)[25:30])
                score_list_test['up_down_precision'].append(np.reshape(topk, -1)[30:35])
                score_list_test['up_down_recall'].append(np.reshape(topk, -1)[35:40])
                score_list_test['up_down_f1score'].append(np.reshape(topk, -1)[40:45])
                
                print(f"CV\t{fold}\tEpoch\tTest\t{epoch+1}\tF1score\t{f1score}")
                print(f"CV\t{fold}\tEpoch\tTest\t{epoch+1}\tUp Precision\t{np.reshape(topk, -1)[0:5]}")
                print(f"CV\t{fold}\tEpoch\tTest\t{epoch+1}\tUp Recall\t{np.reshape(topk, -1)[5:10]}")
                print(f"CV\t{fold}\tEpoch\tTest\t{epoch+1}\tUp F1score\t{np.reshape(topk, -1)[10:15]}")
                print(f"CV\t{fold}\tEpoch\tTest\t{epoch+1}\tDown Precision\t{np.reshape(topk, -1)[15:20]}")
                print(f"CV\t{fold}\tEpoch\tTest\t{epoch+1}\tDown Recall\t{np.reshape(topk, -1)[20:25]}")
                print(f"CV\t{fold}\tEpoch\tTest\t{epoch+1}\tDown F1score\t{np.reshape(topk, -1)[25:30]}")
                print(f"CV\t{fold}\tEpoch\tTest\t{epoch+1}\tUp_Down Precision\t{np.reshape(topk, -1)[30:35]}")
                print(f"CV\t{fold}\tEpoch\tTest\t{epoch+1}\tUp_Down Recall\t{np.reshape(topk, -1)[35:40]}")
                print(f"CV\t{fold}\tEpoch\tTest\t{epoch+1}\tUp_Down F1score\t{np.reshape(topk, -1)[40:45]}")
                
                if best_score < f1score:
                    best_score = f1score
                    torch.save({'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict()},
                            'saved_model/Graph_Transformer/%s_%s_%d.ckpt' % (model_name + '_' + loss_type + '_' + label_type ,  \
                                                                    cell_type, fold))
                    
        best_train_epoch = np.argmax(score_list_train['f1score'])
        best_test_epoch = np.argmax(score_list_test['f1score'])
        
        print(f"CV\t{fold}\tBest Epoch on TRAIN\tTrain\t{best_train_epoch+1}\tF1score\t{score_list_train['f1score'][best_train_epoch]}")
        print(f"CV\t{fold}\tBest Epoch on TRAIN\tTrain\t{best_train_epoch+1}\tUp Precision\t{score_list_train['up_precision'][best_train_epoch]}")
        print(f"CV\t{fold}\tBest Epoch on TRAIN\tTrain\t{best_train_epoch+1}\tUp Recall\t{score_list_train['up_recall'][best_train_epoch]}")
        print(f"CV\t{fold}\tBest Epoch on TRAIN\tTrain\t{best_train_epoch+1}\tUp F1score\t{score_list_train['up_f1score'][best_train_epoch]}")
        print(f"CV\t{fold}\tBest Epoch on TRAIN\tTrain\t{best_train_epoch+1}\tDown Precision\t{score_list_train['down_precision'][best_train_epoch]}")
        print(f"CV\t{fold}\tBest Epoch on TRAIN\tTrain\t{best_train_epoch+1}\tDown Recall\t{score_list_train['down_recall'][best_train_epoch]}")
        print(f"CV\t{fold}\tBest Epoch on TRAIN\tTrain\t{best_train_epoch+1}\tDown F1score\t{score_list_train['down_f1score'][best_train_epoch]}")
        print(f"CV\t{fold}\tBest Epoch on TRAIN\tTrain\t{best_train_epoch+1}\tUp_Down Precision\t{score_list_train['up_down_precision'][best_train_epoch]}")
        print(f"CV\t{fold}\tBest Epoch on TRAIN\tTrain\t{best_train_epoch+1}\tUp_Down Recall\t{score_list_train['up_down_recall'][best_train_epoch]}")
        print(f"CV\t{fold}\tBest Epoch on TRAIN\tTrain\t{best_train_epoch+1}\tUp_Down F1score\t{score_list_train['up_down_f1score'][best_train_epoch]}")

        print(f"CV\t{fold}\tBest Epoch on TRAIN\tTest\t{best_train_epoch+1}\tF1score\t{score_list_test['f1score'][best_train_epoch]}")
        print(f"CV\t{fold}\tBest Epoch on TRAIN\tTest\t{best_train_epoch+1}\tUp Precision\t{score_list_test['up_precision'][best_train_epoch]}")
        print(f"CV\t{fold}\tBest Epoch on TRAIN\tTest\t{best_train_epoch+1}\tUp Recall\t{score_list_test['up_recall'][best_train_epoch]}")
        print(f"CV\t{fold}\tBest Epoch on TRAIN\tTest\t{best_train_epoch+1}\tUp F1score\t{score_list_test['up_f1score'][best_train_epoch]}")
        print(f"CV\t{fold}\tBest Epoch on TRAIN\tTest\t{best_train_epoch+1}\tDown Precision\t{score_list_test['down_precision'][best_train_epoch]}")
        print(f"CV\t{fold}\tBest Epoch on TRAIN\tTest\t{best_train_epoch+1}\tDown Recall\t{score_list_test['down_recall'][best_train_epoch]}")
        print(f"CV\t{fold}\tBest Epoch on TRAIN\tTest\t{best_train_epoch+1}\tDown F1score\t{score_list_test['down_f1score'][best_train_epoch]}")
        print(f"CV\t{fold}\tBest Epoch on TRAIN\tTest\t{best_train_epoch+1}\tUp_Down Precision\t{score_list_test['up_down_precision'][best_train_epoch]}")
        print(f"CV\t{fold}\tBest Epoch on TRAIN\tTest\t{best_train_epoch+1}\tUp_Down Recall\t{score_list_test['up_down_recall'][best_train_epoch]}")
        print(f"CV\t{fold}\tBest Epoch on TRAIN\tTest\t{best_train_epoch+1}\tUp_Down F1score\t{score_list_test['up_down_f1score'][best_train_epoch]}")
        
        print(f"CV\t{fold}\tBest Epoch on TEST\tTrain\t{best_test_epoch+1}\tF1score\t{score_list_train['f1score'][best_test_epoch]}")
        print(f"CV\t{fold}\tBest Epoch on TEST\tTrain\t{best_test_epoch+1}\tUp Precision\t{score_list_train['up_precision'][best_test_epoch]}")
        print(f"CV\t{fold}\tBest Epoch on TEST\tTrain\t{best_test_epoch+1}\tUp Recall\t{score_list_train['up_recall'][best_test_epoch]}")
        print(f"CV\t{fold}\tBest Epoch on TEST\tTrain\t{best_test_epoch+1}\tUp F1score\t{score_list_train['up_f1score'][best_test_epoch]}")
        print(f"CV\t{fold}\tBest Epoch on TEST\tTrain\t{best_test_epoch+1}\tDown Precision\t{score_list_train['down_precision'][best_test_epoch]}")
        print(f"CV\t{fold}\tBest Epoch on TEST\tTrain\t{best_test_epoch+1}\tDown Recall\t{score_list_train['down_recall'][best_test_epoch]}")
        print(f"CV\t{fold}\tBest Epoch on TEST\tTrain\t{best_test_epoch+1}\tDown F1score\t{score_list_train['down_f1score'][best_test_epoch]}")
        print(f"CV\t{fold}\tBest Epoch on TEST\tTrain\t{best_test_epoch+1}\tUp_Down Precision\t{score_list_train['up_down_precision'][best_test_epoch]}")
        print(f"CV\t{fold}\tBest Epoch on TEST\tTrain\t{best_test_epoch+1}\tUp_Down Recall\t{score_list_train['up_down_recall'][best_test_epoch]}")
        print(f"CV\t{fold}\tBest Epoch on TEST\tTrain\t{best_test_epoch+1}\tUp_Down F1score\t{score_list_train['up_down_f1score'][best_test_epoch]}")

        print(f"CV\t{fold}\tBest Epoch on TEST\tTest\t{best_test_epoch+1}\tF1score\t{score_list_test['f1score'][best_test_epoch]}")
        print(f"CV\t{fold}\tBest Epoch on TEST\tTest\t{best_test_epoch+1}\tUp Precision\t{score_list_test['up_precision'][best_test_epoch]}")
        print(f"CV\t{fold}\tBest Epoch on TEST\tTest\t{best_test_epoch+1}\tUp Recall\t{score_list_test['up_recall'][best_test_epoch]}")
        print(f"CV\t{fold}\tBest Epoch on TEST\tTest\t{best_test_epoch+1}\tUp F1score\t{score_list_test['up_f1score'][best_test_epoch]}")
        print(f"CV\t{fold}\tBest Epoch on TEST\tTest\t{best_test_epoch+1}\tDown Precision\t{score_list_test['down_precision'][best_test_epoch]}")
        print(f"CV\t{fold}\tBest Epoch on TEST\tTest\t{best_test_epoch+1}\tDown Recall\t{score_list_test['down_recall'][best_test_epoch]}")
        print(f"CV\t{fold}\tBest Epoch on TEST\tTest\t{best_test_epoch+1}\tDown F1score\t{score_list_test['down_f1score'][best_test_epoch]}")
        print(f"CV\t{fold}\tBest Epoch on TEST\tTest\t{best_test_epoch+1}\tUp_Down Precision\t{score_list_test['up_down_precision'][best_test_epoch]}")
        print(f"CV\t{fold}\tBest Epoch on TEST\tTest\t{best_test_epoch+1}\tUp_Down Recall\t{score_list_test['up_down_recall'][best_test_epoch]}")
        print(f"CV\t{fold}\tBest Epoch on TEST\tTest\t{best_test_epoch+1}\tUp_Down F1score\t{score_list_test['up_down_f1score'][best_test_epoch]}") 
            
    end_time = datetime.now()
    print(end_time-start_time)

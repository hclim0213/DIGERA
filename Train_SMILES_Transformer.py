import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from utils import data_selection, CustomDataset, custom_top_k_score, custom_score
from model import Transformer 

import argparse
from datetime import datetime
from tqdm import tqdm

import warnings 

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

seed = 42
np.random.seed(seed=seed)
torch.manual_seed(seed)

warnings.filterwarnings("ignore")
    
if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='Transformer')
    parser.add_argument('--cell_type', help='MCF7/HA1E/A375...etc')
    # parser.add_argument('--model_name', help='SMILES Transformer/SMILES LSTM')
    parser.add_argument('--loss_type', help='torch_mse/list_wise_rankcosine')
    parser.add_argument('--batch_size', help='Data Number/Train')
    parser.add_argument('--max_epoch', help='Number of Iterations')
    parser.add_argument('--drop_out', help='float')
    parser.add_argument('--lr', help='learning_rate')

    parser.add_argument('--data_file', help='X data path')
    parser.add_argument('--meta_file', help='meta data path, dosage and time condition')
    parser.add_argument('--target_file', help='y data path')
    
    args = parser.parse_args()
    
    cell_type = args.cell_type
    # model_name = args.model_name
    loss_type = args.loss_type
    batch_size = int(args.batch_size)
    max_epoch = int(args.max_epoch)
    drop_out = float(args.drop_out)
    lr = float(args.lr)

    data_file = args.data_file
    meta_file = args.meta_file
    target_file = args.target_file 

    model = 'Transformer'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    """
    X = data_file
    meta = meta_file
    y = target_file
    label <- from target_file 
    
    """
    X, meta, y, label = data_selection(data_file, meta_file, target_file)

    input_dim = X.shape[1:]
    output_dim = y.shape[1]
    meta_dim = meta.shape[1]
    
    start_time = datetime.now()
    dataset = CustomDataset(X, meta, y)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    for i, (train_idx, test_idx) in enumerate(skf.split(dataset.x, label)):        
        train_data = torch.utils.data.Subset(dataset, train_idx)
        test_data = torch.utils.data.Subset(dataset, test_idx)
        
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader  = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=0)

        # metrics 
        best_epoch = 0
        best_score = -1 
        
        score_list_train = {'f1score':[], 'up_precision':[],      'up_recall':[],      'up_f1score':[], \
                            'down_precision':[],    'down_recall':[],    'down_f1score':[], \
                            'up_down_precision':[], 'up_down_recall':[], 'up_down_f1score':[], \
                         }
    
        score_list_test = {'f1score':[], 'up_precision':[],      'up_recall':[],      'up_f1score':[], \
                            'down_precision':[],    'down_recall':[],    'down_f1score':[], \
                            'up_down_precision':[], 'up_down_recall':[], 'up_down_f1score':[], \
                         }

        model = Transformer(input_dim=input_dim, d_model=64, output_dim=output_dim, meta_dim = meta_dim,
                            loss_type=loss_type, nhead=1, d_hid=1024, nlayers=1, dropout=drop_out, device=device).to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        for epoch in tqdm(range(max_epoch)):
            model.train()
            for batch in train_loader:
                x = batch[0].to(device)
                cond = batch[1].to(device)
                y = batch[2].to(device)
                
                optimizer.zero_grad()
                output = model(x, cond)

                # print(x.shape, y.shape, output.shape)
                loss = model.loss(y, output)
                loss.backward()
                
                optimizer.step()
            
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
            
            f1score = custom_score(true_y, pred_rank_np)
            score_list_train['f1score'].append(f1score)

            topk = custom_top_k_score(true_y, pred_rank_np)
            score_list_train['up_precision'].append(np.reshape(topk, -1)[0:5])
            score_list_train['up_recall'].append(np.reshape(topk, -1)[5:10])
            score_list_train['up_f1score'].append(np.reshape(topk, -1)[10:15])
            score_list_train['down_precision'].append(np.reshape(topk, -1)[15:20])
            score_list_train['down_recall'].append(np.reshape(topk, -1)[20:25])
            score_list_train['down_f1score'].append(np.reshape(topk, -1)[25:30])
            score_list_train['up_down_precision'].append(np.reshape(topk, -1)[30:35])
            score_list_train['up_down_recall'].append(np.reshape(topk, -1)[35:40])
            score_list_train['up_down_f1score'].append(np.reshape(topk, -1)[40:45])

            print(f"CV\t{i}\tEpoch\tTrain\t{epoch+1}\tF1score\t{f1score}")
            print(f"CV\t{i}\tEpoch\tTrain\t{epoch+1}\tUp Precision\t{np.reshape(topk, -1)[0:5]}")
            print(f"CV\t{i}\tEpoch\tTrain\t{epoch+1}\tUp Recall\t{np.reshape(topk, -1)[5:10]}")
            print(f"CV\t{i}\tEpoch\tTrain\t{epoch+1}\tUp F1score\t{np.reshape(topk, -1)[10:15]}")
            print(f"CV\t{i}\tEpoch\tTrain\t{epoch+1}\tDown Precision\t{np.reshape(topk, -1)[15:20]}")
            print(f"CV\t{i}\tEpoch\tTrain\t{epoch+1}\tDown Recall\t{np.reshape(topk, -1)[20:25]}")
            print(f"CV\t{i}\tEpoch\tTrain\t{epoch+1}\tDown F1score\t{np.reshape(topk, -1)[25:30]}")
            print(f"CV\t{i}\tEpoch\tTrain\t{epoch+1}\tUp_Down Precision\t{np.reshape(topk, -1)[30:35]}")
            print(f"CV\t{i}\tEpoch\tTrain\t{epoch+1}\tUp_Down Recall\t{np.reshape(topk, -1)[35:40]}")
            print(f"CV\t{i}\tEpoch\tTrain\t{epoch+1}\tUp_Down F1score\t{np.reshape(topk, -1)[40:45]}")
            print('')

            model.eval()
            iter_n = 0

            with torch.no_grad():
                for batch in test_loader:
                    x = batch[0].to(device)
                    cond = batch[1].to(device)
                    y = batch[2].to(device).detach().cpu().numpy()

                    pred_rank = model.predict(x, cond)            
                    if iter_n == 0:
                        pred_rank_np = np.array(pred_rank)
                        true_y = y
                    else:
                        pred_rank_np = np.append(pred_rank_np, np.array(pred_rank), axis=0)
                        true_y = np.append(true_y, y, axis=0)
                    iter_n+=1

            f1score = custom_score(true_y, pred_rank_np)
            score_list_test['f1score'].append(f1score)

            topk = custom_top_k_score(true_y, pred_rank_np)
            score_list_test['up_precision'].append(np.reshape(topk, -1)[0:5])
            score_list_test['up_recall'].append(np.reshape(topk, -1)[5:10])
            score_list_test['up_f1score'].append(np.reshape(topk, -1)[10:15])
            score_list_test['down_precision'].append(np.reshape(topk, -1)[15:20])
            score_list_test['down_recall'].append(np.reshape(topk, -1)[20:25])
            score_list_test['down_f1score'].append(np.reshape(topk, -1)[25:30])
            score_list_test['up_down_precision'].append(np.reshape(topk, -1)[30:35])
            score_list_test['up_down_recall'].append(np.reshape(topk, -1)[35:40])
            score_list_test['up_down_f1score'].append(np.reshape(topk, -1)[40:45])

            print(f"CV\t{i}\tEpoch\tTest\t{epoch+1}\tF1score\t{f1score}")
            print(f"CV\t{i}\tEpoch\tTest\t{epoch+1}\tUp Precision\t{np.reshape(topk, -1)[0:5]}")
            print(f"CV\t{i}\tEpoch\tTest\t{epoch+1}\tUp Recall\t{np.reshape(topk, -1)[5:10]}")
            print(f"CV\t{i}\tEpoch\tTest\t{epoch+1}\tUp F1score\t{np.reshape(topk, -1)[10:15]}")
            print(f"CV\t{i}\tEpoch\tTest\t{epoch+1}\tDown Precision\t{np.reshape(topk, -1)[15:20]}")
            print(f"CV\t{i}\tEpoch\tTest\t{epoch+1}\tDown Recall\t{np.reshape(topk, -1)[20:25]}")
            print(f"CV\t{i}\tEpoch\tTest\t{epoch+1}\tDown F1score\t{np.reshape(topk, -1)[25:30]}")
            print(f"CV\t{i}\tEpoch\tTest\t{epoch+1}\tUp_Down Precision\t{np.reshape(topk, -1)[30:35]}")
            print(f"CV\t{i}\tEpoch\tTest\t{epoch+1}\tUp_Down Recall\t{np.reshape(topk, -1)[35:40]}")
            print(f"CV\t{i}\tEpoch\tTest\t{epoch+1}\tUp_Down F1score\t{np.reshape(topk, -1)[40:45]}")
            print('')

            if best_score < f1score:
                best_score = f1score
                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                            'saved_model/SMILES_Transformer/%s_%s_%d.ckpt' % (model_name + '_' + loss_type,  \
                                                                cell_type, i))
        best_train_epoch = np.argmax(score_list_train['f1score'])
        best_test_epoch = np.argmax(score_list_test['f1score'])
        
        print(f"CV\t{i}\tBest Epoch\tTrain on best train\t{best_train_epoch+1}\tF1score\t{score_list_train['f1score'][best_train_epoch]}")
        print(f"CV\t{i}\tBest Epoch\tTrain on best train\t{best_train_epoch+1}\tUp Precision\t{score_list_train['up_precision'][best_train_epoch]}")
        print(f"CV\t{i}\tBest Epoch\tTrain on best train\t{best_train_epoch+1}\tUp Recall\t{score_list_train['up_recall'][best_train_epoch]}")
        print(f"CV\t{i}\tBest Epoch\tTrain on best train\t{best_train_epoch+1}\tUp F1score\t{score_list_train['up_f1score'][best_train_epoch]}")
        print(f"CV\t{i}\tBest Epoch\tTrain on best train\t{best_train_epoch+1}\tDown Precision\t{score_list_train['down_precision'][best_train_epoch]}")
        print(f"CV\t{i}\tBest Epoch\tTrain on best train\t{best_train_epoch+1}\tDown Recall\t{score_list_train['down_recall'][best_train_epoch]}")
        print(f"CV\t{i}\tBest Epoch\tTrain on best train\t{best_train_epoch+1}\tDown F1score\t{score_list_train['down_f1score'][best_train_epoch]}")
        print(f"CV\t{i}\tBest Epoch\tTrain on best train\t{best_train_epoch+1}\tUp_Down Precision\t{score_list_train['up_down_precision'][best_train_epoch]}")
        print(f"CV\t{i}\tBest Epoch\tTrain on best train\t{best_train_epoch+1}\tUp_Down Recall\t{score_list_train['up_down_recall'][best_train_epoch]}")
        print(f"CV\t{i}\tBest Epoch\tTrain on best train\t{best_train_epoch+1}\tUp_Down F1score\t{score_list_train['up_down_f1score'][best_train_epoch]}")

        print(f"CV\t{i}\tBest Epoch\tTrain on best test\t{best_test_epoch+1}\tF1score\t{score_list_train['f1score'][best_test_epoch]}")
        print(f"CV\t{i}\tBest Epoch\tTrain on best test\t{best_test_epoch+1}\tUp Precision\t{score_list_train['up_precision'][best_test_epoch]}")
        print(f"CV\t{i}\tBest Epoch\tTrain on best test\t{best_test_epoch+1}\tUp Recall\t{score_list_train['up_recall'][best_test_epoch]}")
        print(f"CV\t{i}\tBest Epoch\tTrain on best test\t{best_test_epoch+1}\tUp F1score\t{score_list_train['up_f1score'][best_test_epoch]}")
        print(f"CV\t{i}\tBest Epoch\tTrain on best test\t{best_test_epoch+1}\tDown Precision\t{score_list_train['down_precision'][best_test_epoch]}")
        print(f"CV\t{i}\tBest Epoch\tTrain on best test\t{best_test_epoch+1}\tDown Recall\t{score_list_train['down_recall'][best_test_epoch]}")
        print(f"CV\t{i}\tBest Epoch\tTrain on best test\t{best_test_epoch+1}\tDown F1score\t{score_list_train['down_f1score'][best_test_epoch]}")
        print(f"CV\t{i}\tBest Epoch\tTrain on best test\t{best_test_epoch+1}\tUp_Down Precision\t{score_list_train['up_down_precision'][best_test_epoch]}")
        print(f"CV\t{i}\tBest Epoch\tTrain on best test\t{best_test_epoch+1}\tUp_Down Recall\t{score_list_train['up_down_recall'][best_test_epoch]}")
        print(f"CV\t{i}\tBest Epoch\tTrain on best test\t{best_test_epoch+1}\tUp_Down F1score\t{score_list_train['up_down_f1score'][best_test_epoch]}")

        print(f"CV\t{i}\tBest Epoch\tTest on best train\t{best_train_epoch+1}\tF1score\t{score_list_test['f1score'][best_train_epoch]}")
        print(f"CV\t{i}\tBest Epoch\tTest on best train\t{best_train_epoch+1}\tUp Precision\t{score_list_test['up_precision'][best_train_epoch]}")
        print(f"CV\t{i}\tBest Epoch\tTest on best train\t{best_train_epoch+1}\tUp Recall\t{score_list_test['up_recall'][best_train_epoch]}")
        print(f"CV\t{i}\tBest Epoch\tTest on best train\t{best_train_epoch+1}\tUp F1score\t{score_list_test['up_f1score'][best_train_epoch]}")
        print(f"CV\t{i}\tBest Epoch\tTest on best train\t{best_train_epoch+1}\tDown Precision\t{score_list_test['down_precision'][best_train_epoch]}")
        print(f"CV\t{i}\tBest Epoch\tTest on best train\t{best_train_epoch+1}\tDown Recall\t{score_list_test['down_recall'][best_train_epoch]}")
        print(f"CV\t{i}\tBest Epoch\tTest on best train\t{best_train_epoch+1}\tDown F1score\t{score_list_test['down_f1score'][best_train_epoch]}")
        print(f"CV\t{i}\tBest Epoch\tTest on best train\t{best_train_epoch+1}\tUp_Down Precision\t{score_list_test['up_down_precision'][best_train_epoch]}")
        print(f"CV\t{i}\tBest Epoch\tTest on best train\t{best_train_epoch+1}\tUp_Down Recall\t{score_list_test['up_down_recall'][best_train_epoch]}")
        print(f"CV\t{i}\tBest Epoch\tTest on best train\t{best_train_epoch+1}\tUp_Down F1score\t{score_list_test['up_down_f1score'][best_train_epoch]}")
        
        print(f"CV\t{i}\tBest Epoch\tTest on best test\t{best_test_epoch+1}\tF1score\t{score_list_test['f1score'][best_test_epoch]}")
        print(f"CV\t{i}\tBest Epoch\tTest on best test\t{best_test_epoch+1}\tUp Precision\t{score_list_test['up_precision'][best_test_epoch]}")
        print(f"CV\t{i}\tBest Epoch\tTest on best test\t{best_test_epoch+1}\tUp Recall\t{score_list_test['up_recall'][best_test_epoch]}")
        print(f"CV\t{i}\tBest Epoch\tTest on best test\t{best_test_epoch+1}\tUp F1score\t{score_list_test['up_f1score'][best_test_epoch]}")
        print(f"CV\t{i}\tBest Epoch\tTest on best test\t{best_test_epoch+1}\tDown Precision\t{score_list_test['down_precision'][best_test_epoch]}")
        print(f"CV\t{i}\tBest Epoch\tTest on best test\t{best_test_epoch+1}\tDown Recall\t{score_list_test['down_recall'][best_test_epoch]}")
        print(f"CV\t{i}\tBest Epoch\tTest on best test\t{best_test_epoch+1}\tDown F1score\t{score_list_test['down_f1score'][best_test_epoch]}")
        print(f"CV\t{i}\tBest Epoch\tTest on best test\t{best_test_epoch+1}\tUp_Down Precision\t{score_list_test['up_down_precision'][best_test_epoch]}")
        print(f"CV\t{i}\tBest Epoch\tTest on best test\t{best_test_epoch+1}\tUp_Down Recall\t{score_list_test['up_down_recall'][best_test_epoch]}")
        print(f"CV\t{i}\tBest Epoch\tTest on best test\t{best_test_epoch+1}\tUp_Down F1score\t{score_list_test['up_down_f1score'][best_test_epoch]}")
        
    end_time = datetime.now()
    print(end_time-start_time)

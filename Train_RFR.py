from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from utils import custom_score, custom_top_k_score
from model import CustomRandomForestRegressor
import joblib
    
if __name__=="__main__":
    X_data = pd.read_csv('data/Bayesian_phase2_L5_All_num_features.tsv.gz', sep='\t', compression='gzip')
    y_data = pd.read_csv('data/Bayesian_all_phase2_L5_rank.tsv.gz', sep='\t', compression='gzip')

    rfr_estimator = 2500
    rfr_depth = 20
    # rfr_feature = ['auto']

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
        np_X = X_data[X_data['KNN_labels'].isin(selected_labels_arr)].drop(columns='KNN_labels').iloc[:, 6:].to_numpy()
        np_y = y_data[y_data['KNN_labels'].isin(selected_labels_arr)].drop(columns='KNN_labels').iloc[:, 7:].to_numpy()
        
        selected_labels = X_data[X_data['KNN_labels'].isin(selected_labels_arr)]['KNN_labels'].to_numpy()
        print(f"Cell : {cell} / {selected_labels_arr} \t {selected_labels.shape}")
        print(f"Input shape : {np_X.shape}, {np_y.shape}")

        temp_rfr = 'E'+str(rfr_estimator)+"_D"+str(rfr_depth)
        # cv starts 
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

        n_iter = 0
        best_score = -np.inf

        for i, (train_index, test_index) in enumerate(skf.split(np_X, selected_labels)):
            X_train, X_test   = np_X[train_index, :], np_X[test_index, :]
            y1_train, y1_test = np_y[train_index, :], np_y[test_index, :]
            
            model_rf = CustomRandomForestRegressor(n_estimators=rfr_estimator, max_depth=rfr_depth, \
                                                    random_state=42, verbose=1, n_jobs=-1)
            model_rf.fit(X_train, y1_train)

            # Train process 
            y_train_pred_rank, y_train_pred = model_rf.predict(X_train)
            np.savez(f"output/RFR/RandomForest_cv_{i}_train_{cell}", pred_rank=y_train_pred_rank, pred_median=y_train_pred)
            
            print(selected_labels_arr, "\t", temp_rfr, "\t", "train_cv", n_iter, "\tCustom F1-score\t", custom_score(y1_train, y_train_pred_rank))
            train_topk = custom_top_k_score(y1_train, y_train_pred_rank)
            print(selected_labels_arr, "\t", temp_rfr, "\t", "train_cv", n_iter, "\tCustom Top K Scores\t", "\t".join(np.char.mod('%f', np.reshape(train_topk, -1))))

            print(selected_labels_arr, "\t", temp_rfr, "\t", "train_cv", n_iter, "\tCustom_topk_Up_Precision\t", "\t".join(np.char.mod('%f', np.reshape(train_topk, -1)[0:5])))
            print(selected_labels_arr, "\t", temp_rfr, "\t", "train_cv", n_iter, "\tCustom_topk_Up_Recall\t", "\t".join(np.char.mod('%f', np.reshape(train_topk, -1)[5:10])))
            print(selected_labels_arr, "\t", temp_rfr, "\t", "train_cv", n_iter, "\tCustom_topk_Up_F1score\t", "\t".join(np.char.mod('%f', np.reshape(train_topk, -1)[10:15])))
            print(selected_labels_arr, "\t", temp_rfr, "\t", "train_cv", n_iter, "\tCustom_topk_Down_Precision\t", "\t".join(np.char.mod('%f', np.reshape(train_topk, -1)[15:20])))
            print(selected_labels_arr, "\t", temp_rfr, "\t", "train_cv", n_iter, "\tCustom_topk_Down_Recall\t", "\t".join(np.char.mod('%f', np.reshape(train_topk, -1)[20:25])))
            print(selected_labels_arr, "\t", temp_rfr, "\t", "train_cv", n_iter, "\tCustom_topk_Down_F1score\t", "\t".join(np.char.mod('%f', np.reshape(train_topk, -1)[25:30])))
            print(selected_labels_arr, "\t", temp_rfr, "\t", "train_cv", n_iter, "\tCustom_topk_Up_Down_Precision\t", "\t".join(np.char.mod('%f', np.reshape(train_topk, -1)[30:35])))
            print(selected_labels_arr, "\t", temp_rfr, "\t", "train_cv", n_iter, "\tCustom_topk_Up_Down_Recall\t", "\t".join(np.char.mod('%f', np.reshape(train_topk, -1)[35:40])))
            print(selected_labels_arr, "\t", temp_rfr, "\t", "train_cv", n_iter, "\tCustom_topk_Up_Down_F1score\t", "\t".join(np.char.mod('%f', np.reshape(train_topk, -1)[40:45])))
            
            # Test process 
            y_test_pred_rank, y_test_pred = model_rf.predict(X_test)
            np.savez(f"output/RFR/RandomForest_cv_{i}_test_{cell}", pred_rank=y_test_pred_rank, pred_median=y_test_pred)

            test_f1 = custom_score(y1_test, y_test_pred_rank)
            print(selected_labels_arr, "\t", temp_rfr, "\t", "test_cv", n_iter, "\tCustom F1-score\t", test_f1)
            test_topk = custom_top_k_score(y1_test, y_test_pred_rank)
            print(selected_labels_arr, "\t", temp_rfr, "\t", "test_cv", n_iter, "\tCustom Top K Scores\t", "\t".join(np.char.mod('%f', np.reshape(test_topk, -1))))

            print(selected_labels_arr, "\t", temp_rfr, "\t", "test_cv", n_iter, "\tCustom_topk_Up_Precision\t", "\t".join(np.char.mod('%f', np.reshape(test_topk, -1)[0:5])))
            print(selected_labels_arr, "\t", temp_rfr, "\t", "test_cv", n_iter, "\tCustom_topk_Up_Recall\t", "\t".join(np.char.mod('%f', np.reshape(test_topk, -1)[5:10])))
            print(selected_labels_arr, "\t", temp_rfr, "\t", "test_cv", n_iter, "\tCustom_topk_Up_F1score\t", "\t".join(np.char.mod('%f', np.reshape(test_topk, -1)[10:15])))
            print(selected_labels_arr, "\t", temp_rfr, "\t", "test_cv", n_iter, "\tCustom_topk_Down_Precision\t", "\t".join(np.char.mod('%f', np.reshape(test_topk, -1)[15:20])))
            print(selected_labels_arr, "\t", temp_rfr, "\t", "test_cv", n_iter, "\tCustom_topk_Down_Recall\t", "\t".join(np.char.mod('%f', np.reshape(test_topk, -1)[20:25])))
            print(selected_labels_arr, "\t", temp_rfr, "\t", "test_cv", n_iter, "\tCustom_topk_Down_F1score\t", "\t".join(np.char.mod('%f', np.reshape(test_topk, -1)[25:30])))
            print(selected_labels_arr, "\t", temp_rfr, "\t", "test_cv", n_iter, "\tCustom_topk_Up_Down_Precision\t", "\t".join(np.char.mod('%f', np.reshape(test_topk, -1)[30:35])))
            print(selected_labels_arr, "\t", temp_rfr, "\t", "test_cv", n_iter, "\tCustom_topk_Up_Down_Recall\t", "\t".join(np.char.mod('%f', np.reshape(test_topk, -1)[35:40])))
            print(selected_labels_arr, "\t", temp_rfr, "\t", "test_cv", n_iter, "\tCustom_topk_Up_Down_F1score\t", "\t".join(np.char.mod('%f', np.reshape(test_topk, -1)[40:45])))
            print('')

            joblib.dump(model_rf, f"saved_model/RFR/RandomForest_cv_{i}_{cell}.joblib.gz", compress=('gzip', 3))

            n_iter = n_iter + 1
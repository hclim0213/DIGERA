import numpy as np
from sklearn.metrics import mean_squared_error, roc_auc_score, ndcg_score, average_precision_score
from sklearn.metrics import f1_score, recall_score, confusion_matrix
from scipy.stats import pearsonr, spearmanr, kendalltau, rankdata

def cconfusion_matrix(y_true, y_pred):
    y_true_clip = rank_custom(y_true)
    y_pred_clip = rank_custom(y_pred) 
    
    y_true_round = np.round(y_true_clip*10).astype(int)
    y_pred_round = np.round(y_pred_clip*10).astype(int)
    
    y_true_flat = [item for sublist in y_true_round for item in sublist]
    y_pred_flat = [item for sublist in y_pred_round for item in sublist]
    
    custom_confusion_matrix = confusion_matrix(y_true_flat, y_pred_flat, labels=list(range(11)))
    
    return custom_confusion_matrix 

def rank_custom(y_pred):
    ranks_up = rankdata(y_pred, method='min', axis=1)
    ranks_down = rankdata(-y_pred, method='min', axis=1)

    y_pred_rank = np.where(ranks_down <=1, 1.0,
                  np.where(ranks_down <=10, 0.9,
                  np.where(ranks_down <=50, 0.8,
                  np.where(ranks_down <=100, 0.7,
                  np.where(ranks_down <=200, 0.6,
                  np.where(ranks_up <= 1, 0.0,
                  np.where(ranks_up <= 10, 0.1, 
                  np.where(ranks_up <= 50, 0.2,
                  np.where(ranks_up <= 100, 0.3,
                  np.where(ranks_up <= 200, 0.4, 0.5))))))))))

    return y_pred_rank

def custom_top_k_score(y_true, y_pred):

    y_true_clip = rank_custom(y_true)
    y_pred_clip = rank_custom(y_pred) 
    
    y_true_round = np.round(y_true_clip*10).astype(int)
    y_pred_round = np.round(y_pred_clip*10).astype(int)
    
    y_true_flat = [item for sublist in y_true_round for item in sublist]
    y_pred_flat = [item for sublist in y_pred_round for item in sublist]
    
    custom_confusion_matrix = confusion_matrix(y_true_flat, y_pred_flat, labels=list(range(11)))
    
    Up1_precision =  custom_confusion_matrix[10, 10] / custom_confusion_matrix[:, 10].sum()
    Up1_recall =  custom_confusion_matrix[10, 10] / custom_confusion_matrix[10, :].sum()
    Up1_f1score = 2 * (Up1_precision * Up1_recall) / (Up1_precision + Up1_recall) if (Up1_precision + Up1_recall) != 0 else 0
    
    Down1_precision =  custom_confusion_matrix[0, 0] / custom_confusion_matrix[:, 0].sum()
    Down1_recall =  custom_confusion_matrix[0, 0] / custom_confusion_matrix[0, :].sum()
    Down1_f1score = 2 * (Down1_precision * Down1_recall) / (Down1_precision + Down1_recall) if (Down1_precision + Down1_recall) !=0 else 0
    
    Up_Down1_precision =  (custom_confusion_matrix[0, 0] + custom_confusion_matrix[10, 10]) / (custom_confusion_matrix[:, 0].sum() + custom_confusion_matrix[:, 10].sum())
    Up_Down1_recall =  (custom_confusion_matrix[0, 0] + custom_confusion_matrix[10, 10]) / (custom_confusion_matrix[0, :].sum() + custom_confusion_matrix[10, :].sum())
    Up_Down1_f1score = 2 * (Up_Down1_precision * Up_Down1_recall) / (Up_Down1_precision + Up_Down1_recall) if (Up_Down1_precision + Up_Down1_recall) != 0 else 0
    
    Up10_precision =  custom_confusion_matrix[9:11, 9:11].sum() / custom_confusion_matrix[:, 9:11].sum()
    Up10_recall =  custom_confusion_matrix[9:11, 9:11].sum() / custom_confusion_matrix[9:11, :].sum()
    Up10_f1score = 2 * (Up10_precision * Up10_recall) / (Up10_precision + Up10_recall)
    
    Down10_precision =  custom_confusion_matrix[0:2, 0:2].sum() / custom_confusion_matrix[:, 0:2].sum()
    Down10_recall =  custom_confusion_matrix[0:2, 0:2].sum() / custom_confusion_matrix[0:2, :].sum()
    Down10_f1score = 2 * (Down10_precision * Down10_recall) / (Down10_precision + Down10_recall)
    
    Up_Down10_precision =  (custom_confusion_matrix[0:2, 0:2].sum() + custom_confusion_matrix[9:11, 9:11].sum()) / (custom_confusion_matrix[:, 0:2].sum() + custom_confusion_matrix[:, 9:11].sum())
    Up_Down10_recall =  (custom_confusion_matrix[0:2, 0:2].sum() + custom_confusion_matrix[9:11, 9:11].sum()) / (custom_confusion_matrix[0:2, :].sum() + custom_confusion_matrix[9:11, :].sum())
    Up_Down10_f1score = 2 * (Up_Down10_precision * Up_Down10_recall) / (Up_Down10_precision + Up_Down10_recall)
    
    Up50_precision =  custom_confusion_matrix[8:11, 8:11].sum() / custom_confusion_matrix[:, 8:11].sum()
    Up50_recall =  custom_confusion_matrix[8:11, 8:11].sum() / custom_confusion_matrix[8:11, :].sum()
    Up50_f1score = 2 * (Up50_precision * Up50_recall) / (Up50_precision + Up50_recall)
    
    Down50_precision =  custom_confusion_matrix[0:3, 0:3].sum() / custom_confusion_matrix[:, 0:3].sum()
    Down50_recall =  custom_confusion_matrix[0:3, 0:3].sum() / custom_confusion_matrix[0:3, :].sum()
    Down50_f1score = 2 * (Down50_precision * Down50_recall) / (Down50_precision + Down50_recall)
    
    Up_Down50_precision =  (custom_confusion_matrix[0:3, 0:3].sum() + custom_confusion_matrix[8:11, 8:11].sum()) / (custom_confusion_matrix[:, 0:3].sum() + custom_confusion_matrix[:, 8:11].sum())
    Up_Down50_recall =  (custom_confusion_matrix[0:3, 0:3].sum() + custom_confusion_matrix[8:11, 8:11].sum()) / (custom_confusion_matrix[0:3, :].sum() + custom_confusion_matrix[8:11, :].sum())
    Up_Down50_f1score = 2 * (Up_Down50_precision * Up_Down50_recall) / (Up_Down50_precision + Up_Down50_recall)
        
    Up100_precision =  custom_confusion_matrix[7:11, 7:11].sum() / custom_confusion_matrix[:, 7:11].sum()
    Up100_recall =  custom_confusion_matrix[7:11, 7:11].sum() / custom_confusion_matrix[7:11, :].sum()
    Up100_f1score = 2 * (Up100_precision * Up100_recall) / (Up100_precision + Up100_recall)
    
    Down100_precision =  custom_confusion_matrix[0:4, 0:4].sum() / custom_confusion_matrix[:, 0:4].sum()
    Down100_recall =  custom_confusion_matrix[0:4, 0:4].sum() / custom_confusion_matrix[0:4, :].sum()
    Down100_f1score = 2 * (Down100_precision * Down100_recall) / (Down100_precision + Down100_recall)
    
    Up_Down100_precision =  (custom_confusion_matrix[0:4, 0:4].sum() + custom_confusion_matrix[7:11, 7:11].sum()) / (custom_confusion_matrix[:, 0:4].sum() + custom_confusion_matrix[:, 7:11].sum())
    Up_Down100_recall =  (custom_confusion_matrix[0:4, 0:4].sum() + custom_confusion_matrix[7:11, 7:11].sum()) / (custom_confusion_matrix[0:4, :].sum() + custom_confusion_matrix[7:11, :].sum())
    Up_Down100_f1score = 2 * (Up_Down100_precision * Up_Down100_recall) / (Up_Down100_precision + Up_Down100_recall)
    
    Up200_precision =  custom_confusion_matrix[6:11, 6:11].sum() / custom_confusion_matrix[:, 6:11].sum()
    Up200_recall =  custom_confusion_matrix[6:11, 6:11].sum() / custom_confusion_matrix[6:11, :].sum()
    Up200_f1score = 2 * (Up200_precision * Up200_recall) / (Up200_precision + Up200_recall)
    
    Down200_precision =  custom_confusion_matrix[0:5, 0:5].sum() / custom_confusion_matrix[:, 0:5].sum()
    Down200_recall =  custom_confusion_matrix[0:5, 0:5].sum() / custom_confusion_matrix[0:5, :].sum()
    Down200_f1score = 2 * (Down200_precision * Down200_recall) / (Down200_precision + Down200_recall)
    
    Up_Down200_precision =  (custom_confusion_matrix[0:5, 0:5].sum() + custom_confusion_matrix[6:11, 6:11].sum()) / (custom_confusion_matrix[:, 0:5].sum() + custom_confusion_matrix[:, 6:11].sum())
    Up_Down200_recall =  (custom_confusion_matrix[0:5, 0:5].sum() + custom_confusion_matrix[6:11, 6:11].sum()) / (custom_confusion_matrix[0:5, :].sum() + custom_confusion_matrix[6:11, :].sum())
    Up_Down200_f1score = 2 * (Up_Down200_precision * Up_Down200_recall) / (Up_Down200_precision + Up_Down200_recall)

    Up_Down_array = [[Up_Down1_precision, Up_Down10_precision, Up_Down50_precision, Up_Down100_precision, Up_Down200_precision], 
                 [Up_Down1_recall, Up_Down10_recall, Up_Down50_recall, Up_Down100_recall, Up_Down200_recall], 
                 [Up_Down1_f1score, Up_Down10_f1score, Up_Down50_f1score, Up_Down100_f1score, Up_Down200_f1score]]
    
    return np.around(Up_Down_array, 4)

def custom_top_k_score_2(y_true, y_pred):
    y_true_clip = rank_custom(y_true)
    y_pred_clip = rank_custom(y_pred) 
    
    y_true_round = np.round(y_true_clip*10).astype(int)
    y_pred_round = np.round(y_pred_clip*10).astype(int)
    
    y_true_flat = [item for sublist in y_true_round for item in sublist]
    y_pred_flat = [item for sublist in y_pred_round for item in sublist]
    
    custom_confusion_matrix = confusion_matrix(y_true_flat, y_pred_flat, labels=list(range(11)))
    
    Up1_precision =  custom_confusion_matrix[6:11, 10].sum() / custom_confusion_matrix[:, 10].sum()
    Up1_recall =  custom_confusion_matrix[10, 6:11].sum() / custom_confusion_matrix[10, :].sum()
    Up1_f1score = 2 * (Up1_precision * Up1_recall) / (Up1_precision + Up1_recall)
    
    Down1_precision =  custom_confusion_matrix[0:5, 0].sum() / custom_confusion_matrix[:, 0].sum() \
                        if custom_confusion_matrix[:, 0].sum() != 0 else 0
    Down1_recall =  custom_confusion_matrix[0, 0:5].sum() / custom_confusion_matrix[0, :].sum()
    Down1_f1score = 2 * (Down1_precision * Down1_recall) / (Down1_precision + Down1_recall) \
                        if (Down1_precision + Down1_recall) != 0 else 0
    
    Up_Down1_precision =  (custom_confusion_matrix[0:5, 0].sum() + custom_confusion_matrix[6:11, 10].sum()) / (custom_confusion_matrix[:, 0].sum() + custom_confusion_matrix[:, 10].sum())
    Up_Down1_recall =  (custom_confusion_matrix[0, 0:5].sum() + custom_confusion_matrix[10, 6:11].sum()) / (custom_confusion_matrix[0, :].sum() + custom_confusion_matrix[10, :].sum())
    Up_Down1_f1score = 2 * (Up_Down1_precision * Up_Down1_recall) / (Up_Down1_precision + Up_Down1_recall)
    
    Up10_precision =  custom_confusion_matrix[6:11, 9:11].sum() / custom_confusion_matrix[:, 9:11].sum()
    Up10_recall =  custom_confusion_matrix[9:11, 6:11].sum() / custom_confusion_matrix[9:11, :].sum()
    Up10_f1score = 2 * (Up10_precision * Up10_recall) / (Up10_precision + Up10_recall)
    
    Down10_precision =  custom_confusion_matrix[0:5, 0:2].sum() / custom_confusion_matrix[:, 0:2].sum() \
                        if custom_confusion_matrix[:, 0:2].sum() != 0 else 0 
    Down10_recall =  custom_confusion_matrix[0:2, 0:5].sum() / custom_confusion_matrix[0:2, :].sum()
    Down10_f1score = 2 * (Down10_precision * Down10_recall) / (Down10_precision + Down10_recall) \
                        if (Down10_precision + Down10_recall) != 0 else 0
    
    Up_Down10_precision =  (custom_confusion_matrix[0:5, 0:2].sum() + custom_confusion_matrix[6:11, 9:11].sum()) / (custom_confusion_matrix[:, 0:2].sum() + custom_confusion_matrix[:, 9:11].sum())
    Up_Down10_recall =  (custom_confusion_matrix[0:2, 0:5].sum() + custom_confusion_matrix[9:11, 6:11].sum()) / (custom_confusion_matrix[0:2, :].sum() + custom_confusion_matrix[9:11, :].sum())
    Up_Down10_f1score = 2 * (Up_Down10_precision * Up_Down10_recall) / (Up_Down10_precision + Up_Down10_recall)
    
    Up50_precision =  custom_confusion_matrix[6:11, 8:11].sum() / custom_confusion_matrix[:, 8:11].sum()
    Up50_recall =  custom_confusion_matrix[8:11, 6:11].sum() / custom_confusion_matrix[8:11, :].sum()
    Up50_f1score = 2 * (Up50_precision * Up50_recall) / (Up50_precision + Up50_recall)
    
    Down50_precision =  custom_confusion_matrix[0:5, 0:3].sum() / custom_confusion_matrix[:, 0:3].sum() \
                        if custom_confusion_matrix[:, 0:3].sum() != 0 else 0
    Down50_recall =  custom_confusion_matrix[0:3, 0:5].sum() / custom_confusion_matrix[0:3, :].sum()
    Down50_f1score = 2 * (Down50_precision * Down50_recall) / (Down50_precision + Down50_recall) \
                        if (Down50_precision + Down50_recall) != 0 else 0
    
    Up_Down50_precision =  (custom_confusion_matrix[0:5, 0:3].sum() + custom_confusion_matrix[6:11, 8:11].sum()) / (custom_confusion_matrix[:, 0:3].sum() + custom_confusion_matrix[:, 8:11].sum())
    Up_Down50_recall =  (custom_confusion_matrix[0:3, 0:5].sum() + custom_confusion_matrix[8:11, 6:11].sum()) / (custom_confusion_matrix[0:3, :].sum() + custom_confusion_matrix[8:11, :].sum())
    Up_Down50_f1score = 2 * (Up_Down50_precision * Up_Down50_recall) / (Up_Down50_precision + Up_Down50_recall)
        
    Up100_precision =  custom_confusion_matrix[6:11, 7:11].sum() / custom_confusion_matrix[:, 7:11].sum()
    Up100_recall =  custom_confusion_matrix[7:11, 6:11].sum() / custom_confusion_matrix[7:11, :].sum()
    Up100_f1score = 2 * (Up100_precision * Up100_recall) / (Up100_precision + Up100_recall)
    
    Down100_precision =  custom_confusion_matrix[0:5, 0:4].sum() / custom_confusion_matrix[:, 0:4].sum() \
                            if custom_confusion_matrix[:, 0:4].sum() != 0 else 0
    Down100_recall =  custom_confusion_matrix[0:4, 0:5].sum() / custom_confusion_matrix[0:4, :].sum()
    Down100_f1score = 2 * (Down100_precision * Down100_recall) / (Down100_precision + Down100_recall) \
                            if (Down100_precision + Down100_recall) != 0 else 0
    
    Up_Down100_precision =  (custom_confusion_matrix[0:5, 0:4].sum() + custom_confusion_matrix[6:11, 7:11].sum()) / (custom_confusion_matrix[:, 0:4].sum() + custom_confusion_matrix[:, 7:11].sum())
    Up_Down100_recall =  (custom_confusion_matrix[0:4, 0:5].sum() + custom_confusion_matrix[7:11, 6:11].sum()) / (custom_confusion_matrix[0:4, :].sum() + custom_confusion_matrix[7:11, :].sum())
    Up_Down100_f1score = 2 * (Up_Down100_precision * Up_Down100_recall) / (Up_Down100_precision + Up_Down100_recall)
    
    Up200_precision =  custom_confusion_matrix[6:11, 6:11].sum() / custom_confusion_matrix[:, 6:11].sum()
    Up200_recall =  custom_confusion_matrix[6:11, 6:11].sum() / custom_confusion_matrix[6:11, :].sum()
    Up200_f1score = 2 * (Up200_precision * Up200_recall) / (Up200_precision + Up200_recall)
    
    Down200_precision =  custom_confusion_matrix[0:5, 0:5].sum() / custom_confusion_matrix[:, 0:5].sum() \
                            if custom_confusion_matrix[:, 0:5].sum() != 0 else 0
    Down200_recall =  custom_confusion_matrix[0:5, 0:5].sum() / custom_confusion_matrix[0:5, :].sum()
    Down200_f1score = 2 * (Down200_precision * Down200_recall) / (Down200_precision + Down200_recall) \
                            if (Down200_precision + Down200_recall) != 0 else 0
    
    Up_Down200_precision =  (custom_confusion_matrix[0:5, 0:5].sum() + custom_confusion_matrix[6:11, 6:11].sum()) / (custom_confusion_matrix[:, 0:5].sum() + custom_confusion_matrix[:, 6:11].sum())
    Up_Down200_recall =  (custom_confusion_matrix[0:5, 0:5].sum() + custom_confusion_matrix[6:11, 6:11].sum()) / (custom_confusion_matrix[0:5, :].sum() + custom_confusion_matrix[6:11, :].sum())
    Up_Down200_f1score = 2 * (Up_Down200_precision * Up_Down200_recall) / (Up_Down200_precision + Up_Down200_recall)

    Up_array = [[Up1_precision, Up10_precision, Up50_precision, Up100_precision, Up200_precision], 
                 [Up1_recall, Up10_recall, Up50_recall, Up100_recall, Up200_recall], 
                 [Up1_f1score, Up10_f1score, Up50_f1score, Up100_f1score, Up200_f1score]]

    Down_array = [[Down1_precision, Down10_precision, Down50_precision, Down100_precision, Down200_precision], 
                 [Down1_recall, Down10_recall, Down50_recall, Down100_recall, Down200_recall], 
                 [Down1_f1score, Down10_f1score, Down50_f1score, Down100_f1score, Down200_f1score]]

    Up_Down_array = [[Up_Down1_precision, Up_Down10_precision, Up_Down50_precision, Up_Down100_precision, Up_Down200_precision], 
                 [Up_Down1_recall, Up_Down10_recall, Up_Down50_recall, Up_Down100_recall, Up_Down200_recall], 
                 [Up_Down1_f1score, Up_Down10_f1score, Up_Down50_f1score, Up_Down100_f1score, Up_Down200_f1score]]
    
    return np.round(Up_array, 4), np.round(Down_array, 4), np.round(Up_Down_array, 4)


def rmse(label_test, label_predict):
    return np.sqrt(mean_squared_error(label_test, label_predict))


def correlation(label_test, label_predict, correlation_type):
    if correlation_type == 'pearson':
        corr = pearsonr
    elif correlation_type == 'spearman':
        corr = spearmanr
    else:
        raise ValueError("Unknown correlation type: %s" % correlation_type)
    score = []
    for lb_test, lb_predict in zip(label_test, label_predict):
        score.append(corr(lb_test, lb_predict)[0])
        
    valid_values = np.array([value for value in score if not np.isnan(value)])
    average = np.nanmean(valid_values)
            
    return average
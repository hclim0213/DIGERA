from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr, rankdata
import numpy as np

class CustomRandomForestRegressor(RandomForestRegressor):
    
    def predict(self, X):
        
        y_pred = super().predict(X)
        
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
        
        return y_pred_rank, y_pred
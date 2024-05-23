import numpy as np
import pandas as pd

from rdkit.Chem import MACCSkeys, AllChem
from rdkit import Chem
from rdkit import Chem
from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
import joblib
import argparse 
from pathlib import Path

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import torch 


from MFBERT.Tokenizer.MFBERT_Tokenizer import MFBERTTokenizer
from MFBERT.Model.model import MFBERT

from model import CustomRandomForestRegressor, Transformer, LSTMModel, LinearRegression_3d, Graph_Transformer
from utils import DataPredict, custom_make_rank

dose_dict = {'5um' : [0, 1], '10um' : [1, 0]}
time_dict = {'6h' : [0, 1], '24h' : [1, 0]}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
Structural Descriptors

"""
def smile2fp(smile_list, cell, dose, time, cell_crispr):
    iter_i = 0 
    for temp_smiles in smile_list:
        mol = Chem.MolFromSmiles(temp_smiles)
        temp_maccs_keys = MACCSkeys.GenMACCSKeys(mol).ToList()[:166]
        temp_ECFP6 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToList()
        temp_FCFP4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=1024).ToList()
        temp_PCFP = get_fingerprint(Chem.MolToSmiles(mol), 'pubchem').to_numpy().tolist()
        temp_mfbert = mfbert_fp(temp_smiles)
        cell_fp = cell_crispr[cell_crispr.index == cell].iloc[:, :152].values[0]
        temp_result = np.expand_dims(np.concatenate((temp_maccs_keys, temp_ECFP6, temp_FCFP4, temp_PCFP, temp_mfbert, \
                                 cell_fp, dose_dict[dose], time_dict[time])), axis=0)
        if iter_i == 0:
            result_np = temp_result
        else : 
            result_np = np.append(result_np, temp_result, axis=0)
        iter_i += 1 
        # print(temp_fp_MACCS_ECFP6.shape, temp_fp_MACCS_FCFP4.shape, temp_fp_MACCS_PCFP.shape)
        
    return result_np

def mfbert_fp(smile):
    tokenizer = MFBERTTokenizer.from_pretrained('MFBERT/Tokenizer/Model/', dict_file = 'MFBERT/Tokenizer/Model/dict.txt')
    model = MFBERT(weights_dir='MFBERT/Model/pre-trained', return_attention=False, inference_method='mean')
    mol = Chem.MolFromSmiles(smile)
    temp_mf = model(tokenizer(Chem.MolToSmiles(mol), return_tensors='pt')).detach().numpy()[0].tolist()
    return temp_mf 

"""
SMILES fingerprints 

"""

char_idx = {
            ' ': 0,
            '^': 1,
            '$': 2,
            "#": 20,
            "%": 22,
            "(": 25,
            ")": 24,
            "+": 26,
            "-": 27,
            ".": 30,
            "0": 32,
            "1": 31,
            "2": 34,
            "3": 33,
            "4": 36,
            "5": 35,
            "6": 38,
            "7": 37,
            "8": 40,
            "9": 39,
            "=": 41,
            "A": 7,
            "B": 11,
            "C": 19,
            "F": 4,
            "H": 6,
            "I": 5,
            "N": 10,
            "O": 9,
            "P": 12,
            "S": 13,
            "X": 15,
            "Y": 14,
            "Z": 3,
            "[": 16,
            "]": 18,
            "b": 21,
            "c": 8,
            "n": 17,
            "o": 29,
            "p": 23,
            "s": 28,
            "@": 42,
            "R": 43,
            "/": 44,
            "\\": 45,
            "E": 46,
            "Q": 47,
            "U": 48,
            "V":49,
        }

encode_dict = {"Pt":"Q", "Hg":"U", "Mg":"V", "Br": "Y", "Cl": "X", "Si": "A", "Se": "Z", "@@": "R", "se": "E"}

def encode(smiles: str) -> str:
    temp_smiles = smiles
    for symbol, token in encode_dict.items():
        temp_smiles = temp_smiles.replace(symbol, token)
    return temp_smiles

def smiles_encoder( smiles, maxlen=120 ):
    X = np.zeros( ( maxlen, len( char_idx ) ) )
    for i, c in enumerate( smiles ):
        X[i, char_idx[c] ] = 1
    return X

def smile_fingerprint(smile_input, cell, dose, time, cell_crispr):
    temp_encode = []
    for smile in smile_input.iloc[:, 0]:
        smiles_np = smiles_encoder(encode(smile))
        temp_encode.append(smiles_np)
    encode_np = np.array(temp_encode, dtype=np.int32)

    cell_fp = cell_crispr[cell_crispr.index == cell].iloc[:, :152].values[0]
    fp_np = np.concatenate((cell_fp, dose_dict[dose], time_dict[time]))
    fp_np = np.tile(fp_np, (smile_input.shape[0], 1))
    return encode_np, fp_np

class TotalPredict(object):
    def __init__(self, smile_input, cell, dose, time, crispr_file, device):
        self.cell = cell
        self.dose = dose 
        self.time = time 
        self.device = device
        self.crispr_file = crispr_file
        self.smile_input = smile_input
        self.encode_np, self.fp_np = smile_fingerprint(smile_input, cell, dose, time, crispr_file)
        
        self.model_config={'input_dim' : self.encode_np.shape[1:],
                'ntoken' : self.encode_np.shape[2],
                'output_dim' : 978,
                'meta_dim' : self.fp_np.shape[1], 
                'transformer_n_layers' : 1, 
                'transformer_d_model': 64,
                'transformer_d_hid' : 1024,
                'transformer_head' : 1, 
                'transformer_loss' : 'list_wise_rankcosine',
                'lstm_n_layers' : 1, 
                'lstm_d_model': 512, 
                'lstm_d_hid': 128,
                'lstm_loss' : 'point_wise_mse', 
                'ciger_n_layers': 2,
                'ciger_n_heads' : 1, 
                'ciger_loss' : 'point_wise_mse', 
            }

    def RandomForest_predict(self, fold):
        mult_smile = smile2fp(self.smile_input.iloc[:, 0], self.cell, self.dose, self.time, self.crispr_file)
        file_path = f"saved_model/RFR/RandomForest_cv_{fold}_{self.cell}.joblib.gz"
        temp_model = joblib.load(file_path)
        temp_model.set_params(verbose=0)

        rf_rank = temp_model.predict(mult_smile)

        return rf_rank

    def Transformer_predict(self, fold):
        model = Transformer(input_dim=self.model_config['input_dim'], d_model=self.model_config['transformer_d_model'], 
                            output_dim=self.model_config['output_dim'], meta_dim=self.model_config['meta_dim'],
                            loss_type=self.model_config['transformer_loss'], nhead=1, d_hid=self.model_config['transformer_d_hid'], 
                            nlayers=self.model_config['transformer_n_layers'], dropout=0.01, device=device).to(device)
        checkpoint = torch.load(f"saved_model/SMILES_Transformer/"+
                            f"Transformer_{self.model_config['transformer_loss']}_rank_{self.cell}_{fold}.ckpt")
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
    
        encode_np = torch.tensor(self.encode_np, dtype=torch.int32).to(self.device)
        fp_np = torch.tensor(self.fp_np, dtype=torch.float32).to(self.device)
    
        trans_rank = model.predict(encode_np, fp_np)
    
        return trans_rank 
    
    def LSTM_predict(self, fold):
        model = LSTMModel(input_dim=self.model_config['input_dim'], d_model=self.model_config['lstm_d_model'], 
                      output_dim=self.model_config['output_dim'], meta_dim = self.model_config['meta_dim'], 
                      loss_type=self.model_config['lstm_loss'], d_hid=self.model_config['lstm_d_hid'], 
                      nlayers=self.model_config['lstm_n_layers'], dropout=0.01, device=device).to(device)
        checkpoint = torch.load(f"saved_model/SMILES_LSTM"+ \
                        f"LSTM_{self.model_config['lstm_loss']}_rank_{self.cell}_{fold}.ckpt")
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        encode_np = torch.tensor(self.encode_np, dtype=torch.int32).to(self.device)
        fp_np = torch.tensor(self.fp_np, dtype=torch.float32).to(self.device)

        
        hidden = model.init_hidden(self.model_config['ntoken']) # batch_size but size error 
        lstm_rank = model.predict(encode_np, fp_np, hidden)
    
        return lstm_rank
    
    def GTF_predict(self, gene_file, fold):
        data = DataPredict(smile_input, gene_file, self.time, self.cell, self.dose, self.device)
        fp_type = 'neural'
        label_type = 'real'
        initializer = torch.nn.init.xavier_uniform_
        
        gtf = Graph_Transformer(drug_input_dim=data.drug_dim, 
              gene_embed=data.gene, 
              gene_input_dim=data.gene.size()[1],
              n_layers=self.model_config['ciger_n_layers'], n_heads=self.model_config['ciger_n_heads'], 
              encode_dim=512, 
              fp_type=fp_type, 
              loss_type=self.model_config['ciger_loss'], 
              label_type=label_type, 
              device=self.device,
              initializer=initializer, 
              pert_type_input_dim=data.pert_type_dim, 
              cell_id_input_dim=data.cell_id_dim,
              pert_idose_input_dim=data.pert_idose_dim, 
              use_pert_type=data.use_pert_type,
              use_cell_id=data.use_cell_id, 
              use_pert_idose=data.use_pert_idose)
        
        checkpoint = torch.load(f"saved_model/Graph_Transformer/Transformer_{self.model_config['ciger_loss']}_real_{self.cell}_{fold}.ckpt")
        gtf.load_state_dict(checkpoint['model_state_dict'])
        gtf.to(device)
        gtf.eval()
        
        output = data.get_feature_from_drug()
        rank = gtf.predict(output['drug'], output['gene'], output['pert_type'], \
                                             output['cell_id'], output['pert_idose'])
        return rank

def validate_tsv(filename):
    """
    Custom type function for argparse - checks if the file ends with .tsv
    """
    if not filename.endswith('.tsv'):
        raise argparse.ArgumentTypeError("File must be a .tsv file")
    return filename


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Ensemble Lasso prediction from SMILES')
    parser.add_argument('--cell', help='MCF7, VCAP, PC3, etc')
    parser.add_argument('--input_file', type=validate_tsv, help='input file')
    parser.add_argument('--output_file', type=validate_tsv, help='output tsv file')
    parser.add_argument('--gene_file', help='Gene file for Graph Transformer')
    parser.add_argument('--crispr_file', help='Cell viability file of CRISPR')

    args = parser.parse_args()
    cell = args.cell
    input_file = args.input_file
    output_file = args.output_file 
    gene_file = args.gene_file
    crispr_file = args.crispr_file
    
    dose_list = list(dose_dict.keys())
    time_list = list(time_dict.keys())

    col = ['cell_id', 'pert_idose', 'pert_itime']
    col = col + ['gen' + str(num+1) for num in range(978)]

    smile_input = pd.read_csv(input_file, sep='\t', header=None)     
    fp_file = Path(output_file)
    
    final_df = pd.DataFrame(columns=col)
    for dose in dose_list:
        for time in time_list:
            temp_dict = dict()
            for i in range(5):
                predict=TotalPredict(smile_input, cell, dose, time, crispr_file, device)
                rf_output = predict.RandomForest_predict(fold=i)
                tf_output = predict.Transformer_predict(fold=i)
                lstm_output = predict.LSTM_predict(fold=i)
                gtf_output = predict.GTF_predict(gene_file, fold=i)
                
                rank_stack = np.stack((rf_output, tf_output, lstm_output, gtf_output), axis=2)
                meta_model = joblib.load(f"saved_model/Ensemble/Lasso_cv_{i}_{cell}.pkl")

                rank = meta_model.predict(rank_stack)
                rank = rank.squeeze()

                for num, index in enumerate(smile_input[1]):
                    if i == 0:
                        temp_dict[index] = rank[num]
                    else:
                        temp_dict[index] = np.vstack((temp_dict[index], rank[num]))
            
            result_dict = dict()
            for key, item in temp_dict.items():
                result_dict[key] = item.mean(axis=0)
            temp_result = pd.DataFrame.from_dict(data=result_dict, orient='index')
            temp_result = custom_make_rank(temp_result)
            temp_result = pd.DataFrame(temp_result, columns = [ 'gen'+str(num+1) for num in range(978) ], index=list(result_dict.keys()))
            temp_result.insert(0, 'pert_itime', time, allow_duplicates=False)   
            temp_result.insert(0, 'pert_idose', dose, allow_duplicates=False)
            temp_result.insert(0, 'cell_id', cell, allow_duplicates=False)
            temp_result.insert(0, 'smiles', list(smile_input[0]), allow_duplicates=False)
            if fp_file.is_file():
                temp_result.to_csv(fp_file, header=False, sep='\t', mode='a')
            else:
                temp_result.to_csv(fp_file, header=True, sep='\t', mode='a')

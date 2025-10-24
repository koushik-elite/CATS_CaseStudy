import pandas as pd
import numpy as np
from exp.exp_main import Exp_Main
from data.data_loader import Dataset_Custom
import torch
import warnings

from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

class NiftyPredictor:
    def __init__(self, seq_len=96, pred_len=96):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.features = 'MS'  # Multivariate input, univariate output
        self.model = 'CATSF'
        
    def prepare_data(self, file_path):
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Convert datetime
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        # Normalize the data using sklearn's StandardScaler
        self.scaler = StandardScaler()
        scaled_data = self.scaler.fit_transform(df.values)
        df_scaled = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)

        return df_scaled
    
    def create_exp(self):
        # Hard-coded experiment arguments as requested
        args = type('Args', (), {
            'is_training': 0,
            'model_id': 'nift_96',
            'model': self.model,
            'data': 'pred',
            'root_path': 'dataset/nift/',
            'data_path': 'NIFTY_COMMODITIES_30T_VALID.csv',
            'features': 'M',
            'target': 'close',
            'freq': 'h',
            'checkpoints': './checkpoints/',
            'inverse': True,
            'ratios': '0.7,0.1,0.2',
            'seq_len': 96,
            'label_len': 48,
            'pred_len': 96,
            'dec_in': 4,
            'd_model': 256,
            'n_heads': 32,
            'e_layers': 0,
            'd_layers': 3,
            'd_ff': 512,
            'dropout': 0.2,
            'embed': 'timeF',
            'use_norm': 1,
            'num_workers': 10,
            'itr': 1,
            'train_epochs': 30,
            'batch_size': 16,
            'patience': 10,
            'learning_rate': 0.001,
            'des': 'Exp',
            'loss': 'MSE',
            'lradj': 'type2',
            'use_amp': False,
            'use_gpu': True,
            'gpu': 0,
            'use_multi_gpu': False,
            'devices': '0,1,2,3',
            'test_flop': False,
            'QAM_start': 0.1,
            'QAM_end': 0.2,
            'patch_len': 24,
            'stride': 24,
            'pct_start': 0.3,
            'padding_patch': 'end',
            'query_independence': False,
            'store_attn': False,
            'enc_in': 4,
            'c_out': 1,
        })()
        
        return Exp_Main(args)

    def predict(self, file_path, model_path):
        # Prepare data
        df = self.prepare_data(file_path)
        
        # Create experiment
        exp = self.create_exp()
        
        # Load model
        exp.model.load_state_dict(torch.load(model_path), map_location='cuda:0')

        # Prepare input data
        input_data = df.values[-self.seq_len:]
        print("Input data shape:", input_data.shape)
        input_data = torch.FloatTensor(input_data).unsqueeze(0)
        print("Input data tensor shape:", input_data.shape)
        # Make prediction
        exp.model.eval()
        with torch.no_grad():
            output = exp.model(input_data)
            
        # Inverse transform predictions
        predictions = self.scaler.inverse_transform(output.squeeze().numpy())
        
        # Create prediction dates
        last_date = df.index[-1]
        pred_dates = pd.date_range(start=last_date, periods=self.pred_len+1, freq='30T')[1:]
        
        # Create prediction dataframe
        pred_df = pd.DataFrame({
            'datetime': pred_dates,
            'predicted_close': predictions[:, 0]  # assuming close is the first column
        })
        
        return pred_df

def main():
    # Initialize predictor
    predictor = NiftyPredictor(seq_len=96, pred_len=96)
    
    # File paths
    data_file = '/home/koushik/CATS_CaseStudy/dataset/nift/NIFTY_COMMODITIES_30T_VALID.csv'
    model_file = '/home/koushik/CATS_CaseStudy/checkpoints/nift_96_192_CATSF_custom_sl96_pl192_dm256_nh32_dl3_df512_qiFalse_0/checkpoint.pth'
    
    # Make predictions
    predictions = predictor.predict(data_file, model_file)
    
    # Save predictions
    predictions.to_csv('nifty_predictions.csv', index=False)
    print("Predictions saved to nifty_predictions.csv")
    
    # Display first few predictions
    print("\nFirst few predictions:")
    print(predictions.head())

if __name__ == "__main__":
    main()
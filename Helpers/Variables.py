import os
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


CLASS_NUM = 3

DATA_DIR = '/opt/workspace/MS_data/BIO'

DATASET_DIR = '/opt/workspace/Seohyeon/TTA/data'

WD = f'/opt/workspace/Seohyeon/SOSO_App' # 'or' WD = os.getcwd()
RD = '/opt/workspace/Seohyeon/SOSO_App'

METRICS = ['loss', 'acc', 'bacc', 'f1', 'preci', 'recall']

def Make_COL_NAMES(bool):
    if bool == True:
        COL_NAMES = [
            'Fp1', 'Fp2', 'AF3', 'AF4', 'F7', 'F8', 'F3', 'Fz', 'F4', 'FC5', 
            'FC6', 'T7', 'T8', 'C3', 'C4', 'CP5', 'CP6', 'P7', 'P8', 'P3', 
            'Pz', 'P4', 'PO7', 'PO8', 'PO3', 'PO4', 'O1', 'O2', 
            'ECG', 'Resp', 'PPG', 'GSR','ACC_LAT','ACC_LONG','YAW_RATE'
        ]
    else:
        COL_NAMES = [
            'Fp1', 'Fp2', 'AF3', 'AF4', 'F7', 'F8', 'F3', 'Fz', 'F4', 'FC5', 
            'FC6', 'T7', 'T8', 'C3', 'C4', 'CP5', 'CP6', 'P7', 'P8', 'P3', 
            'Pz', 'P4', 'PO7', 'PO8', 'PO3', 'PO4', 'O1', 'O2', 
            'ECG', 'Resp', 'PPG', 'GSR'
        ]
    return COL_NAMES

# 파일명
FILENAME_MODEL = f'MS__model.pt'
FILENAME_HIST = f'MS__history.csv'
FILENAME_HISTSUM = f'MS__history_summary.csv'
FILENAME_RES = f'MS__result.csv'

FILENAME_TOTALSUM = f'MS__total_summary.csv'
FILENAME_TOTALRES = f'MS__total_result.csv' 
FILENAME_FOLDSUM = f'MS__fold_summary.csv'

FILENAME_TOTALSUM_E1 = f'MS__EXPT1_total_summary.csv'
FILENAME_TOTALSUM_E2 = f'MS__EXPT2_total_summary.csv'

FILENAME_TOTALRES_E1 = f'MS__EXPT1_total_result.csv'
FILENAME_TOTALRES_E2 = f'MS__EXPT1_total_result.csv'

FILENAME_FOLDSUM_E1 = f'MS__EXPT1_fold_summary.csv'
FILENAME_FOLDSUM_E2 = f'MS__EXPT2_fold_summary.csv'

FILENAME_FOLDSUM_ALL = f'MS__ALL_fold_summary.csv'
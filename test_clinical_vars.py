#!/usr/bin/env python3

from dlbcl.utils.data_loader import load_common_data
import pandas as pd

# Load data for patho2
print("Loading patho2 data...")
data = load_common_data('patho2')

if data:
    print('\nClinical data columns:')
    for col in sorted(data['clinical_data'].columns):
        print(f'  - {col}')
    
    print(f'\nMerged data columns with clinical variables:')
    clinical_cols = [col for col in data['merged_data'].columns 
                     if not col.startswith('feature_') and col != 'patient_id']
    for col in sorted(clinical_cols):
        non_null_count = data['merged_data'][col].notna().sum()
        total_count = len(data['merged_data'])
        print(f'  - {col}: {non_null_count}/{total_count} non-null values')
        
    print(f'\nTotal merged samples: {len(data["merged_data"])}')
    print(f'Total feature columns: {sum(1 for col in data["merged_data"].columns if col.startswith("feature_"))}')
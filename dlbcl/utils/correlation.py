"""
Correlation analysis utility functions
Used by correlation.py and module.py
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist

from .data_loader import load_common_data


def load_both_datasets():
    """Load both morph and patho2 datasets for comparison analysis"""
    morph_data_dict = load_common_data('morph')
    patho2_data_dict = load_common_data('patho2')
    
    if morph_data_dict is None or patho2_data_dict is None:
        raise RuntimeError("Failed to load dataset(s). Check data availability.")
    
    return morph_data_dict['merged_data'], patho2_data_dict['merged_data']


def get_clinical_mapping(patho2_columns):
    """Get clinical variable mapping based on patho2 naming convention"""
    if 'CD10 IHC' in patho2_columns:
        # New naming convention
        return {
            'CD10 IHC': 'CD10 IHC',
            'MUM1 IHC': 'MUM1 IHC',
            'BCL2 IHC': 'BCL2 IHC',
            'MYC IHC': 'MYC IHC',
            'BCL6 IHC': 'BCL6 IHC',
            'HANS': 'HANS'
        }
    else:
        # Old naming convention
        return {
            'CD10 IHC': 'CD10',
            'MUM1 IHC': 'MUM1',
            'BCL2 IHC': 'BCL2',
            'MYC IHC': 'MYC',
            'BCL6 IHC': 'BCL6',
            'HANS': 'HANS'
        }


def compute_correlation_matrix(merged_data, correlation_method='pearson'):
    """Compute correlation matrix between features and clinical variables"""
    
    # Get clinical columns (maintaining original order)
    clinical_cols = [col for col in merged_data.columns if not col.startswith('feature_') and col != 'patient_id']
    feature_cols = [col for col in merged_data.columns if col.startswith('feature_')]
    
    print(f"Computing correlations: {len(feature_cols)} features x {len(clinical_cols)} clinical variables")
    
    # Create correlation matrix
    correlation_matrix = np.full((len(feature_cols), len(clinical_cols)), np.nan)
    
    for i, feature_col in enumerate(feature_cols):
        for j, clinical_col in enumerate(clinical_cols):
            # Get valid (non-NaN) pairs
            feature_values = merged_data[feature_col]
            clinical_values = merged_data[clinical_col]
            
            # Create mask for valid pairs
            valid_mask = ~(feature_values.isna() | clinical_values.isna())
            
            if valid_mask.sum() < 10:  # Need at least 10 valid pairs
                continue
                
            valid_feature = feature_values[valid_mask]
            valid_clinical = clinical_values[valid_mask]
            
            # Compute correlation
            if correlation_method == 'pearson':
                corr, _ = pearsonr(valid_feature, valid_clinical)
            elif correlation_method == 'spearman':
                corr, _ = spearmanr(valid_feature, valid_clinical)
            else:
                raise ValueError(f"Unknown correlation method: {correlation_method}")
            
            correlation_matrix[i, j] = corr
    
    # Convert to DataFrame
    corr_df = pd.DataFrame(correlation_matrix, 
                          index=feature_cols,
                          columns=clinical_cols)
    
    return corr_df


def prepare_comparison_matrices(morph_corr, patho2_corr, clinical_mapping):
    """Prepare data matrices for comparison analysis"""
    
    # Create comprehensive variable list combining both datasets
    all_variables = []
    
    # Add common variables
    for morph_var, patho2_var in clinical_mapping.items():
        if morph_var in morph_corr.columns and patho2_var in patho2_corr.columns:
            all_variables.append((morph_var, patho2_var, 'common'))
    
    # Add Morph-only variables
    morph_only = ['OS', 'PFS', 'Follow-up Status', 'Age', 'LDH', 'ECOG PS',
                 'Stage', 'IPI Score', 'IPI Risk Group (4 Class)', 'RIPI Risk Group']
    for var in morph_only:
        if var in morph_corr.columns:
            all_variables.append((var, None, 'morph_only'))
    
    # Add Patho2-only variables
    patho2_only = ['EBV']
    for var in patho2_only:
        if var in patho2_corr.columns:
            all_variables.append((None, var, 'patho2_only'))
    
    # Prepare data matrices
    morph_clinical_vars = []
    patho2_clinical_vars = []
    variable_labels = []
    
    for morph_var, patho2_var, var_type in all_variables:
        if var_type == 'common':
            morph_clinical_vars.append(morph_var)
            patho2_clinical_vars.append(patho2_var)
            variable_labels.append(morph_var)
        elif var_type == 'morph_only':
            morph_clinical_vars.append(morph_var)
            patho2_clinical_vars.append(None)
            variable_labels.append(f"{morph_var}\n(Morph only)")
        elif var_type == 'patho2_only':
            morph_clinical_vars.append(None)
            patho2_clinical_vars.append(patho2_var)
            variable_labels.append(f"{patho2_var}\n(Patho2 only)")
    
    # Create data matrices handling None values
    morph_data = {}
    patho2_data = {}
    
    for i, (morph_var, patho2_var) in enumerate(zip(morph_clinical_vars, patho2_clinical_vars)):
        col_name = f"col_{i}"
        
        if morph_var:
            morph_data[col_name] = morph_corr[morph_var]
        else:
            morph_data[col_name] = pd.Series([np.nan] * len(morph_corr), index=morph_corr.index)
        
        if patho2_var:
            patho2_data[col_name] = patho2_corr[patho2_var]
        else:
            patho2_data[col_name] = pd.Series([np.nan] * len(patho2_corr), index=patho2_corr.index)
    
    morph_subset = pd.DataFrame(morph_data)
    patho2_subset = pd.DataFrame(patho2_data)
    
    return morph_subset, patho2_subset, variable_labels


def create_unified_dendrogram(morph_subset, patho2_subset):
    """Create unified dendrogram using common variables from both datasets"""
    
    # Use common variables for fair dendrogram creation
    n_common = len([col for col in morph_subset.columns if not morph_subset[col].isna().all()])
    
    # Use both datasets' common variables for more robust clustering
    morph_common_only = morph_subset.iloc[:, :n_common].fillna(0)
    patho2_common_only = patho2_subset.iloc[:, :n_common].fillna(0)
    
    # Average correlation patterns from both datasets for fairness
    combined_common = (morph_common_only + patho2_common_only) / 2
    
    feature_distance = pdist(combined_common.values, metric='euclidean')
    feature_linkage = linkage(feature_distance, method='ward')
    feature_order = leaves_list(feature_linkage)
    
    return feature_linkage, feature_order
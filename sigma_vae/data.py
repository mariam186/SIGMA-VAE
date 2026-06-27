"""
Data Processing for Multi-Modal Neuroimaging
Exact same logic - just cleaned up.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler
from typing import Dict, List, Tuple, Optional
import pickle


class DataProcessor:
    """
    Processes multi-modal neuroimaging data from separate CSV files.
    Handles subject alignment, modality-specific preprocessing, and scaling.
    """
    
    def __init__(
        self, 
        output_dir: str = './output',
        scaler_type: str = 'robust'
    ):
        self.output_dir = output_dir
        self.scaler_type = scaler_type
        self.scalers = {}
        self.feature_names = {}
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Metadata columns to drop
        self.metadata_columns = [
            'subject_id', 'age', 'age_normalized', 'sex', 'site', 'specific_diagnosis','dataset_name', 'diagnosis'
        ]
        
        # 
        # Modality-specific columns to drop
        self.modality_drops = {
            'subcortical': [
                'non-WM-hypointensities', '3rd-Ventricle', '4th-Ventricle',
                'eTIV', '5th-Ventricle', 'EstimatedTotalIntraCranialVol'
            ],
            'cortical': [],
            'surface': []
        }
    
    def _load_and_clean_csv(self, path: str, modality: str) -> pd.DataFrame:
        """Load and clean a CSV file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        
        print(f"[{modality}] Loading: {path}")
        df = pd.read_csv(path, low_memory=False)
        
        # DROP specific_diagnosis FIRST (before dropna)
        df = df.drop(columns='specific_diagnosis', errors='ignore')
        print(f"[{modality}] Loaded shape: {df.shape}")
        
        # Standardize subject_id
        if 'subject_id' in df.columns:
            df['subject_id'] = df['subject_id'].astype(str)
        
        # Standardize sex encoding
        if 'sex' in df.columns and df['sex'].dtype == object:
            df['sex'] = df['sex'].map({
                'female': 0, 'male': 1,
                'F': 0, 'M': 1,
                'f': 0, 'm': 1
            })
        
        # Normalize age
        if 'age' in df.columns and 'age_normalized' not in df.columns:
            df['age_normalized'] = df['age'] / 100.0
        
        # THEN drop NaN rows
        df.dropna(inplace=True)
        print(f"[{modality}] After cleaning: {df.shape}")
        
        return df
    
    def _extract_features(self, df: pd.DataFrame, modality: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Extract features from dataframe."""
        # Columns to drop
        columns_to_drop = self.metadata_columns.copy()
        if modality in self.modality_drops:
            columns_to_drop.extend(self.modality_drops[modality])
        
        existing_drops = [col for col in columns_to_drop if col in df.columns]
        
        # Extract metadata
        metadata_cols = ['subject_id', 'age', 'age_normalized', 'sex', 'dataset_name', 'diagnosis']
        existing_meta = [col for col in metadata_cols if col in df.columns]
        metadata_df = df[existing_meta].copy()
        
        # Extract features
        features_df = df.drop(columns=existing_drops, errors='ignore')
        features_df = features_df.select_dtypes(include=[np.number])
        features_df = features_df.loc[:, (features_df != 0).any(axis=0)]
        
        print(f"[{modality}] Features shape: {features_df.shape}")
        print(f"[{modality}] Dropped columns: {existing_drops}")
        
        return features_df, metadata_df
    
    def _align_subjects(self, dataframes: Dict[str, pd.DataFrame]) -> Tuple[Dict, List[str]]:
        """Align subjects across modalities."""
        subject_sets = [set(df['subject_id']) for df in dataframes.values()]
        common_subjects = sorted(list(set.intersection(*subject_sets)))
        
        if len(common_subjects) == 0:
            raise ValueError("No common subjects found!")
        
        print(f"\n✓ Found {len(common_subjects)} common subjects")
        
        aligned_dfs = {}
        for mod_name, df in dataframes.items():
            df_aligned = df[df['subject_id'].isin(common_subjects)].copy()
            df_aligned = df_aligned.sort_values('subject_id').reset_index(drop=True)
            df_aligned = df_aligned.drop_duplicates('subject_id').reset_index(drop=True)
            aligned_dfs[mod_name] = df_aligned
        
        # Verify alignment
        ref_subjects = aligned_dfs[list(dataframes.keys())[0]]['subject_id'].tolist()
        for mod_name, df in aligned_dfs.items():
            if df['subject_id'].tolist() != ref_subjects:
                raise ValueError(f"Alignment failed for {mod_name}")
        
        print("✓ All modalities aligned")
        return aligned_dfs, common_subjects
    
    def load_train_data(
        self,
        data_paths: Dict[str, str],
        val_split: float = 0.15,
        random_state: int = 42
    ) -> Dict:
        """Load training data."""
        print("\n" + "="*70)
        print("LOADING TRAINING DATA")
        print("="*70)
        
        # Load CSVs
        raw_dfs = {}
        for mod_name, path in data_paths.items():
            raw_dfs[mod_name] = self._load_and_clean_csv(path, mod_name)
        
        # Align subjects
        aligned_dfs, common_subjects = self._align_subjects(raw_dfs)
        
        # Extract features
        features_dict = {}
        metadata_dict = {}
        for mod_name, df in aligned_dfs.items():
            features_df, metadata_df = self._extract_features(df, mod_name)
            features_dict[mod_name] = features_df
            metadata_dict[mod_name] = metadata_df
            self.feature_names[mod_name] = features_df.columns.tolist()
        
        # Get targets from first modality
        ref_modality = list(data_paths.keys())[0]
        ref_metadata = metadata_dict[ref_modality]
        age = ref_metadata['age_normalized'].values.astype(np.float32)
        sex = ref_metadata['sex'].values.astype(np.float32)
        
        print(f"\nTargets:")
        print(f"  Age range: {age.min():.2f} - {age.max():.2f}")
        print(f"  Sex distribution: {sex.mean():.2%} male")
        
        # Scale features
        scaled_features = {}
        for mod_name, features_df in features_dict.items():
            if self.scaler_type == 'robust':
                scaler = RobustScaler()
            else:
                scaler = StandardScaler()
            
            scaled = scaler.fit_transform(features_df.values).astype(np.float32)
            self.scalers[mod_name] = scaler
            scaled_features[mod_name] = scaled
            
            print(f"\n[{mod_name}] Scaled: {scaled.shape}, range [{scaled.min():.2f}, {scaled.max():.2f}]")
        
        # Train/val split
        n_total = len(common_subjects)
        n_val = int(n_total * val_split)
        
        np.random.seed(random_state)
        indices = np.random.permutation(n_total)
        train_idx = indices[n_val:]
        val_idx = indices[:n_val]
        
        train_data = {name: data[train_idx] for name, data in scaled_features.items()}
        val_data = {name: data[val_idx] for name, data in scaled_features.items()}
        
        print(f"\nSplit: {len(train_idx)} train, {len(val_idx)} val")
        print("✓ Training data loaded")
        
        return {
            'train_data': train_data,
            'val_data': val_data,
            'age_train': age[train_idx],
            'age_val': age[val_idx],
            'sex_train': sex[train_idx],
            'sex_val': sex[val_idx],
            'train_subjects': [common_subjects[i] for i in train_idx],
            'val_subjects': [common_subjects[i] for i in val_idx],
            'metadata': {
                'modalities': list(data_paths.keys()),
                'feature_dims': {name: data.shape[1] for name, data in train_data.items()},
                'n_train': len(train_idx),
                'n_val': len(val_idx)
            }
        }
    
    def load_test_data(self, data_paths: Dict[str, str]) -> Dict:
        """Load test data using fitted scalers."""
        print("\n" + "="*70)
        print("LOADING TEST DATA")
        print("="*70)
        
        raw_dfs = {}
        for mod_name, path in data_paths.items():
            raw_dfs[mod_name] = self._load_and_clean_csv(path, mod_name)
        
        aligned_dfs, common_subjects = self._align_subjects(raw_dfs)
        
        features_dict = {}
        metadata_dict = {}
        for mod_name, df in aligned_dfs.items():
            features_df, metadata_df = self._extract_features(df, mod_name)
            features_dict[mod_name] = features_df
            metadata_dict[mod_name] = metadata_df
        
        ref_modality = list(data_paths.keys())[0]
        ref_metadata = metadata_dict[ref_modality]
        age = ref_metadata['age_normalized'].values.astype(np.float32)
        sex = ref_metadata['sex'].values.astype(np.float32)
        diagnosis=df['diagnosis'].astype(str)
        dataset_name=df['dataset_name'].astype(str)
        
        # Scale using fitted scalers
        scaled_features = {}
        for mod_name, features_df in features_dict.items():
            if mod_name not in self.scalers:
                raise ValueError(f"No scaler for {mod_name}")
            scaled = self.scalers[mod_name].transform(features_df.values).astype(np.float32)
            scaled_features[mod_name] = scaled
            print(f"[{mod_name}] Test shape: {scaled.shape}")
        
        print("✓ Test data loaded")
        
        return {
            'data': scaled_features,
            'age': age,
            'sex': sex,
            'diagnosis':diagnosis,
            'dataset_name':dataset_name,
            'subject_id': common_subjects
        }
    
    def save(self, path: Optional[str] = None):
        """Save processor state (scalers and feature names)."""
        if path is None:
            path = os.path.join(self.output_dir, 'processor.pkl')
        with open(path, 'wb') as f:
            pickle.dump({
                'scalers': self.scalers,
                'feature_names': self.feature_names,
                'scaler_type': self.scaler_type
            }, f)
        print(f"Processor saved: {path}")
    
    def load(self, path: str):
        """Load processor state."""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.scalers = state['scalers']
        self.feature_names = state['feature_names']
        self.scaler_type = state.get('scaler_type', 'robust')
        print(f"Processor loaded: {path}")


class MultiModalDataset(Dataset):
    """PyTorch Dataset for multi-modal data."""
    
    def __init__(self, data_dict: Dict[str, np.ndarray], age: np.ndarray, sex: np.ndarray):
        self.modalities = list(data_dict.keys())
        self.data = {mod: torch.FloatTensor(data_dict[mod]) for mod in self.modalities}
        self.age = torch.FloatTensor(age)
        self.sex = torch.FloatTensor(sex)
        
        lengths = [len(self.data[mod]) for mod in self.modalities]
        assert len(set(lengths)) == 1
        assert len(self.age) == lengths[0]
    
    def __len__(self):
        return len(self.age)
    
    def __getitem__(self, idx):
        item = {mod: self.data[mod][idx] for mod in self.modalities}
        item['age'] = self.age[idx]
        item['sex'] = self.sex[idx]
        return item


def create_dataloaders(data_dict: Dict, batch_size: int = 64) -> Tuple[DataLoader, DataLoader]:
    """Create train and val dataloaders."""
    train_dataset = MultiModalDataset(
        data_dict['train_data'],
        data_dict['age_train'],
        data_dict['sex_train']
    )
    val_dataset = MultiModalDataset(
        data_dict['val_data'],
        data_dict['age_val'],
        data_dict['sex_val']
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\nDataLoaders: {len(train_loader)} train batches, {len(val_loader)} val batches")
    return train_loader, val_loader

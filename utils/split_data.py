
import os
import glob
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from common import seed_everything

def create_splits(data_dir='Brain_Stroke_CT_Dataset', output_csv='splits.csv'):
    seed_everything(42)
    
    # 1. Gather all file paths
    # DIRECTORY STRUCTURE CHECK:
    # Root/Brain_Stroke_CT_Dataset/Normal/PNG/img.png OR Root/Brain_Stroke_CT_Dataset/Normal/img.png
    
    classes = ['Bleeding', 'Ischemia', 'Normal']
    data = []
    
    # We use os.path.abspath to be safe regardless of where script is run
    if not os.path.exists(data_dir):
        # Try looking one level up if run from utils/
        if os.path.exists(os.path.join('..', data_dir)):
            data_dir = os.path.join('..', data_dir)
            print(f"[INFO] Found dataset at {data_dir}")
    
    print(f"[INFO] Scanning {data_dir}...")
    
    for cls in classes:
        cls_path = os.path.join(data_dir, cls)
        # 1. Try direct images (Folder/*.png)
        files = glob.glob(os.path.join(cls_path, '*.png'))
        # 2. If empty, try subfolder (Folder/PNG/*.png or Folder/*/*.png)
        if not files:
             files = glob.glob(os.path.join(cls_path, '*', '*.png'))
        
        print(f"  - {cls}: Found {len(files)} images")
        
        for f in files:
            # Binary Label: Normal=0, Stroke=1 (Bleeding/Ischemia)
            label = 0 if cls == 'Normal' else 1
            
            # CRITICAL: We stratify on the 'subtype' (Bleeding vs Ischemia) 
            # to ensure the model doesn't learn only one type of stroke in Train.
            stratify_label = cls 
            
            data.append({
                'path': f,
                'binary_label': label,
                'stratify_label': stratify_label
            })
            
    df = pd.DataFrame(data)
    print(f"[INFO] Total Dataset Size: {len(df)}")
    
    # 2. Split Strategy: 70% Train, 15% Val, 15% Test
    # Step A: Split 70% Train vs 30% "Temp" (to be re-split)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    for train_idx, temp_idx in sss1.split(df, df['stratify_label']):
        train_df = df.iloc[train_idx].copy()
        temp_df = df.iloc[temp_idx].copy()
        
    # Step B: Split the 30% "Temp" into equal halves (15% Val, 15% Test)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    for val_idx, test_idx in sss2.split(temp_df, temp_df['stratify_label']):
        val_df = temp_df.iloc[val_idx].copy()
        test_df = temp_df.iloc[test_idx].copy()
        
    # 4. Assign Split Names
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'
    
    # 5. Combine and Save
    final_df = pd.concat([train_df, val_df, test_df])
    
    # 5. Combine
    final_df = pd.concat([train_df, val_df, test_df])
    
    # 6. Normalize Paths
    # Ensure paths are relative to the project root (without ../)
    # E.g., Brain_Stroke_CT_Dataset/Normal/101.png
    def clean_path(p):
        return os.path.relpath(p, start=os.getcwd() if 'utils' not in os.getcwd() else os.path.dirname(os.getcwd()))
        
    # Just ensure forward slashes for cross-platform compatibility if needed, 
    # but Windows python handles mixed well mostly. 
    # Let's just save relative to the dataset PARENT directory.
    # If script runs in root, data_dir is Brain_Stroke_CT_Dataset.
    
    final_df['path'] = final_df['path'].apply(lambda x: x.replace('\\', '/')) 
    
    # 7. Save
    
    final_df.to_csv(output_csv, index=False)
    print(f"[INFO] Splits saved to {output_csv}")
    print(final_df['split'].value_counts())
    print("\nClass Distribution per Split:")
    print(final_df.groupby(['split', 'stratify_label']).size())

if __name__ == "__main__":
    create_splits()

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import os
from torch.nn.utils.rnn import pad_sequence # Useful for padding

# Function defined at the module top-level for pickling
def repeat_channels_if_needed(x):
    if x.ndim == 3 and x.shape[0] == 1:
        return x.repeat(3, 1, 1)
    return x

class MRNetDataset(Dataset):
    def __init__(self, root_dir, task, plane, split='train', transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.task = task
        self.plane = plane
        self.split = split
        self.transform = transform

        self.csv_path = os.path.join(self.root_dir, f'{self.split}-{self.task}.csv')
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}.")
        self.data_info = pd.read_csv(self.csv_path, header=None, names=['id', 'label'])

        self.image_dir = os.path.join(self.root_dir, self.split, self.plane)

        self.available_cases = []
        for i in range(len(self.data_info)):
            case_id = str(self.data_info.iloc[i, 0]).zfill(4)
            image_path = os.path.join(self.image_dir, f'{case_id}.npy')
            if os.path.exists(image_path):
                self.available_cases.append(self.data_info.iloc[i])
        self.data_info = pd.DataFrame(self.available_cases)
        if len(self.data_info) == 0:
            print(f"Warning: No available cases found for {split} split, task {task}, plane {plane}. "
                  f"Checked image_dir: {self.image_dir} and csv: {self.csv_path}")

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        case_id = str(self.data_info.iloc[idx, 0]).zfill(4)
        label = torch.tensor(self.data_info.iloc[idx, 1], dtype=torch.float)

        image_path = os.path.join(self.image_dir, f'{case_id}.npy')
        volume = np.load(image_path) # Shape: (num_slices, height, width)
        volume = volume.astype(np.float32)

        min_val, max_val = volume.min(), volume.max()
        if max_val > 1.0 or min_val < 0.0 :
             if max_val - min_val > 0:
                 volume = (volume - min_val) / (max_val - min_val)
             else:
                 volume = np.zeros_like(volume) if min_val == 0 else np.ones_like(volume) * min_val

        # (num_slices, 1, height, width)
        volume = np.expand_dims(volume, axis=1)

        transformed_slices = []
        for i in range(volume.shape[0]): # Iterate over slices
            slice_img = volume[i] # Shape: (1, H, W)
            slice_tensor = torch.from_numpy(slice_img).float()
            if self.transform:
                transformed_slice = self.transform(slice_tensor)
            else:
                transformed_slice = slice_tensor
            transformed_slices.append(transformed_slice)
        
        # volume_tensor shape: (num_slices, C_out, H_out, W_out)
        # C_out should be 3 after repeat_channels_if_needed and Normalize
        volume_tensor = torch.stack(transformed_slices)
        
        return {'volume': volume_tensor, 'label': label, 'id': case_id}

def pad_collate_fn(batch):
    """
    Collate function to handle varying numbers of slices by padding.
    Args:
        batch: List of samples, where each sample is a dict {'volume': tensor, 'label': tensor, 'id': str}
               'volume' tensor shape: (num_slices, C, H, W)
    Returns:
        A batch dictionary:
            'volumes': padded tensor (batch_size, max_num_slices, C, H, W)
            'labels': tensor (batch_size)
            'ids': list of ids
            'num_slices': list of actual number of slices for each volume in the batch
    """
    # Sort the batch by the number of slices in descending order (optional, but can be useful)
    # batch.sort(key=lambda x: x['volume'].shape[0], reverse=True)

    volumes = [item['volume'] for item in batch]
    labels = torch.stack([item['label'] for item in batch]) # Labels are single values, can be stacked
    ids = [item['id'] for item in batch]
    num_slices = [vol.shape[0] for vol in volumes] # Store original number of slices

    # Pad volumes to the max number of slices in this batch
    # `pad_sequence` expects a list of tensors (L, *), pads along L (dim 0)
    # We need to make volumes (num_slices, C*H*W) or handle 4D padding carefully.
    # Simpler: find max_slices, then pad each volume manually.
    
    max_num_slices = max(num_slices)
    
    padded_volumes = []
    for vol in volumes:
        current_slices, C, H, W = vol.shape
        padding_needed = max_num_slices - current_slices
        if padding_needed > 0:
            # Create padding tensor (padding_needed, C, H, W) with zeros
            padding = torch.zeros((padding_needed, C, H, W), dtype=vol.dtype, device=vol.device)
            padded_vol = torch.cat((vol, padding), dim=0)
        else:
            padded_vol = vol
        padded_volumes.append(padded_vol)
        
    # Stack all padded volumes along a new batch dimension
    padded_volumes_tensor = torch.stack(padded_volumes, dim=0)

    return {
        'volume': padded_volumes_tensor, # Renamed from 'volumes' to match existing key in train.py
        'label': labels,
        'id': ids,
        # 'num_slices': num_slices # You can include this if your model needs it
    }


def get_dataloaders(root_dir, task, plane, batch_size=8, num_workers=4):
    IMG_SIZE = 224
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomRotation(degrees=15),
        transforms.Lambda(repeat_channels_if_needed),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Lambda(repeat_channels_if_needed),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    train_dataset = MRNetDataset(root_dir=root_dir, task=task, plane=plane, split='train', transform=train_transform)
    if len(train_dataset) == 0:
        raise ValueError(f"Training dataset for task '{task}', plane '{plane}' is empty.")

    valid_dataset = MRNetDataset(root_dir=root_dir, task=task, plane=plane, split='valid', transform=val_transform)
    if len(valid_dataset) == 0:
        raise ValueError(f"Validation dataset for task '{task}', plane '{plane}' is empty.")

    # Use the custom collate_fn
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True, collate_fn=pad_collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, 
                              num_workers=num_workers, pin_memory=True, collate_fn=pad_collate_fn)
    
    return train_loader, valid_loader, train_dataset

if __name__ == '__main__':
    # Keep the dummy data setup for testing
    dummy_root = "MRNet-v1.0_dummy" 
    os.makedirs(os.path.join(dummy_root, "train", "sagittal"), exist_ok=True)
    os.makedirs(os.path.join(dummy_root, "valid", "sagittal"), exist_ok=True)

    pd.DataFrame({'id': ['0000', '0001', '0004'], 'label': [0, 1, 0]}).to_csv(os.path.join(dummy_root, 'train-acl.csv'), header=False, index=False)
    pd.DataFrame({'id': ['0002', '0003'], 'label': [1, 0]}).to_csv(os.path.join(dummy_root, 'valid-acl.csv'), header=False, index=False)

    # Create dummy npy files with varying number of slices
    dummy_scan_10_slices = np.random.rand(10, 256, 256).astype(np.float32) # Fewer slices
    dummy_scan_15_slices = np.random.rand(15, 256, 256).astype(np.float32) # More slices
    dummy_scan_12_slices = np.random.rand(12, 256, 256).astype(np.float32)

    np.save(os.path.join(dummy_root, "train", "sagittal", "0000.npy"), dummy_scan_10_slices)
    np.save(os.path.join(dummy_root, "train", "sagittal", "0001.npy"), dummy_scan_15_slices)
    np.save(os.path.join(dummy_root, "train", "sagittal", "0004.npy"), dummy_scan_12_slices) # Added a third sample for better batch testing
    np.save(os.path.join(dummy_root, "valid", "sagittal", "0002.npy"), dummy_scan_10_slices)
    np.save(os.path.join(dummy_root, "valid", "sagittal", "0003.npy"), dummy_scan_15_slices)
    
    print("Dummy files with varying slices created for testing dataloader.")

    try:
        # Test with num_workers=0 first for easier debugging
        train_loader_test, valid_loader_test, _ = get_dataloaders(root_dir=dummy_root, task='acl', plane='sagittal', batch_size=2, num_workers=0) 
        
        print(f"Number of training batches: {len(train_loader_test)}")
        print(f"Number of validation batches: {len(valid_loader_test)}")

        for i, batch in enumerate(train_loader_test):
            print(f"\nTrain Batch {i+1}:")
            # Key for volume is 'volume' as per pad_collate_fn output
            print("Volume shape:", batch['volume'].shape) # Expected: (batch_size, max_slices_in_batch, 3, IMG_SIZE, IMG_SIZE)
            print("Label shape:", batch['label'].shape)
            print("Labels:", batch['label'])
            print("ID:", batch['id'])
            # print("Num Slices:", batch['num_slices']) # If you uncomment this in collate_fn
            if batch['volume'].shape[2] != 3: # Check channel dimension
                 raise ValueError(f"Channel mismatch: Expected 3, got {batch['volume'].shape[2]}")
            if i == 0: 
                break
        
        print("\nDataloader test with num_workers=0 and pad_collate_fn successful.")

        # Test with num_workers > 0 if the above works
        if torch.cuda.is_available() or os.name != 'nt':
            print("\nTesting with num_workers=1 (safer for initial MP test)...") # Reduced to 1 for initial test
            train_loader_mp_test, _, _ = get_dataloaders(root_dir=dummy_root, task='acl', plane='sagittal', batch_size=2, num_workers=1)
            for i, batch_mp in enumerate(train_loader_mp_test):
                print(f"Train Batch MP {i+1} Volume shape:", batch_mp['volume'].shape)
                if i == 0:
                    break
            print("Dataloader test with num_workers=1 and pad_collate_fn successful.")
        else:
            print("Skipping num_workers > 0 test on Windows without __main__ guard or no CUDA.")


    except Exception as e:
        print(f"Error during dataloader test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        import shutil
        # shutil.rmtree(dummy_root) # Comment out to inspect dummy files
        # print(f"Cleaned up dummy directory: {dummy_root}")
        pass

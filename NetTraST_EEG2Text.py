import os
import mne
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import yaml
import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix 
import seaborn as sns 
import matplotlib.pyplot as plt 
from tqdm import tqdm

# Set up device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class Metrics:
    def __init__(self, column_names):
        column_names.insert(0, "time_stamp")
        self.df = pd.DataFrame(columns=column_names)

    def add_row(self, row_list):
        row_list.insert(0, str(datetime.datetime.now()))
        self.df.loc[len(self.df)] = row_list

    def save_to_csv(self, filepath):
        self.df.to_csv(filepath, index=False)

def create_save_dir(save_dir):
    save_dir = os.path.join(save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    return save_dir

class NetTraST(nn.Module):
    def __init__(self, args): 
        super(NetTraST, self).__init__()
        # Input shape: [batch, channels (64), time_points (128)]
        self.batch_norm1 = nn.BatchNorm1d(args['embed_dim'])  # Normalize along channels
        
        # Spatial pathway (processing channels)
        self.spatial_conv = nn.Conv1d(
            in_channels=args['embed_dim'], 
            out_channels=args['kernel_num'], 
            kernel_size=args['kernel_size'], 
            padding=args['kernel_size']//2
        )
        self.spatial_tra = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=args['kernel_num'],
                nhead=args['nhead'],
                dim_feedforward=args['dim_feedforward'],
            ),
            num_layers=3,
        )
        
        # Temporal pathway (processing time points)
        self.temporal_conv = nn.Conv1d(
            in_channels=args['vocab_size'], 
            out_channels=args['kernel_num'], 
            kernel_size=args['kernel_size'], 
            padding=args['kernel_size']//2
        )
        self.temporal_tra = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=args['kernel_num'],
                nhead=args['nhead'],
                dim_feedforward=args['dim_feedforward'],
            ),
            num_layers=3,
        )
        
        # Main transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=args['kernel_num'],
                nhead=args['nhead'],
                dim_feedforward=args['dim_feedforward'],
            ),
            num_layers=args['num_layers'],
        )
        
        # Classification head
        self.batch_norm2 = nn.BatchNorm1d(args['kernel_num'])
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(args['kernel_num'] * args['vocab_size'], args['kernel_num'])
        self.dropout = nn.Dropout(args['dropout'])
        self.fc2 = nn.Linear(args['kernel_num'], args['class_num'])
        self.activation = nn.RReLU(0.1, 0.3)

    def forward(self, x):
        # Input shape: [batch, 64, 128]
        x = self.batch_norm1(x)
        
        # Spatial pathway
        x1 = self.spatial_conv(x)  # [batch, kernel_num, 128]
        x1 = x1.permute(2, 0, 1)  # [128, batch, kernel_num] for transformer
        x1 = self.spatial_tra(x1)
        x1 = x1.permute(1, 2, 0)  # [batch, kernel_num, 128]
        
        # Temporal pathway
        x2 = x.permute(0, 2, 1)  # [batch, 128, 64]
        x2 = self.temporal_conv(x2)  # [batch, kernel_num, 64]
        x2 = x2.permute(2, 0, 1)  # [64, batch, kernel_num] for transformer
        x2 = self.temporal_tra(x2)
        x2 = x2.permute(1, 2, 0)  # [batch, kernel_num, 64]
        
        # Align dimensions for addition
        x2 = nn.functional.interpolate(x2, size=x1.shape[2])  # Upsample to match x1's time dimension
        
        # Combine pathways
        x = x1 + x2  # [batch, kernel_num, 128]
        
        # Final processing
        x = x.permute(2, 0, 1)  # [128, batch, kernel_num]
        x = self.transformer(x)
        x = x.permute(1, 2, 0)  # [batch, kernel_num, 128]
        x = self.batch_norm2(x)
        
        # Classification
        x = self.flatten(x)  # [batch, kernel_num * 128]
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
   
def compute_subject_baseline(raw, events, baseline_mode='eyes_closed'):
    """Compute baseline using eye-open (101) and/or eye-close (102) events"""
    if baseline_mode == 'eyes_open':
        baseline_codes = [101]
    elif baseline_mode == 'eyes_closed':
        baseline_codes = [102]
    else:  # auto or default
        baseline_codes = [101, 102]
    
    baseline_events = [e for e in events if e[2] in baseline_codes]
    if not baseline_events:
        return None
        
    baseline_epochs = mne.Epochs(
        raw, events=np.array(baseline_events), event_id=None,
        tmin=0, tmax=2.0, baseline=None, preload=True
    )
    baseline_data = baseline_epochs.get_data()
    return np.mean(baseline_data, axis=(0, 2))  # [channels]


def load_and_preprocess_data(edf_paths, session_type='read'):
    """
    Parameters:
    -----------
    edf_paths : list
        List of EDF file paths
    session_type : str
        'audio' for imagine speak sessions or 'read' for read sessions
    """
    all_epochs = []
    all_metadata = []
    
    # Define event mappings based on session type
    if session_type == 'audio':
        CLASS_TYPE_RANGES = {
            'bad': '1', 'go': '2', 'good': '3', 'happy': '4', 'hello': '5',
            'help': '6', 'no': '7', 'stop': '8', 'thanks': '9', 'yes': '10'
        }
        baseline_mode = 'eyes_closed'  # For audio sessions
    elif session_type == 'read':
        CLASS_TYPE_RANGES = {
            'bad': (1, 10),
            'go': (11, 20),
            'good': (21, 30),
            'happy': (31, 40),
            'hello': (41, 50),
            'help': (51, 60),
            'no': (61, 70),
            'stop': (71, 80),
            'thanks': (81, 90),
            'yes': (91, 100),
        }
        baseline_mode = 'eyes_open'  # For read sessions
    else:
        raise ValueError("session_type must be either 'audio' or 'read'")

    # Create standardized event IDs (starting from 1000 to avoid conflicts)
    standardized_event_id = {}
    next_event_id = 1000
    
    for edf_path in edf_paths:
        print(f"Loading: {edf_path}")
        raw = mne.io.read_raw_edf(edf_path, preload=True)
        raw.filter(l_freq=1., h_freq=50., fir_design='firwin', verbose=False)
        raw.resample(250)  
        
        events, original_event_id = mne.events_from_annotations(raw)
        baseline_mean = compute_subject_baseline(raw, events, baseline_mode)

        # Build event_id and class mapping differently for each session type
        file_event_id = {}
        class_mapping = {}
        
        if session_type == 'audio':
            for class_name, event_str in CLASS_TYPE_RANGES.items():
                if event_str in original_event_id:
                    # Create standardized key
                    key = f"{class_name}_{event_str}"
                    
                    # Check if we've seen this key before
                    if key not in standardized_event_id:
                        standardized_event_id[key] = next_event_id
                        next_event_id += 1
                    
                    # Use standardized event ID for this file
                    file_event_id[key] = standardized_event_id[key]
                    class_mapping[original_event_id[event_str]] = (class_name, standardized_event_id[key])
        else:  # read session
            for class_name, (start_code, end_code) in CLASS_TYPE_RANGES.items():
                for code in range(start_code, end_code + 1):
                    if str(code) in original_event_id:
                        # Create standardized key
                        key = f"{class_name}_{code}"
                        
                        # Check if we've seen this key before
                        if key not in standardized_event_id:
                            standardized_event_id[key] = next_event_id
                            next_event_id += 1
                        
                        # Use standardized event ID for this file
                        file_event_id[key] = standardized_event_id[key]
                        class_mapping[original_event_id[str(code)]] = (class_name, standardized_event_id[key])

        # Extract valid events and remap them to standardized event IDs
        valid_events = []
        for ev in events:
            if ev[2] in class_mapping:
                class_name, new_event_id = class_mapping[ev[2]]
                new_event = [ev[0], ev[1], new_event_id]
                valid_events.append(new_event)
        
        valid_events = np.array(valid_events)

        if len(valid_events) == 0:
            print(f"Warning: No valid events found in {edf_path}, skipping...")
            continue
        
        # Create task epochs with standardized event IDs
        epochs = mne.Epochs(
            raw, events=valid_events, event_id=file_event_id,
            tmin=0, tmax=0.999, baseline=None, preload=True, verbose=False
        )

        if baseline_mean is not None:
            print(f"Applying baseline correction for {edf_path}")
            epochs._data -= baseline_mean[:, np.newaxis]

        # Create metadata using standardized event IDs
        metadata = pd.DataFrame({
            'event_id': [e[2] for e in epochs.events],
            'class': [next(class_name for orig_id, (class_name, std_id) in class_mapping.items() 
                          if std_id == e[2]) for e in epochs.events],
            'source_file': [edf_path] * len(epochs.events)
        })

        all_epochs.append(epochs)
        all_metadata.append(metadata)
        
        print(f"Loaded {len(epochs)} epochs from {edf_path}")

    if len(all_epochs) == 0:
        raise RuntimeError("No valid data found in provided EDF paths.")
    
    # Now concatenation should work because all epochs use standardized event IDs
    try:
        combined_epochs = mne.concatenate_epochs(all_epochs)
        combined_metadata = pd.concat(all_metadata, ignore_index=True)
        combined_epochs.metadata = combined_metadata
        print(f"Successfully concatenated {len(combined_epochs)} total epochs")
    except Exception as e:
        print(f"Concatenation failed: {e}")
        print("Event IDs in each file:")
        for i, epochs in enumerate(all_epochs):
            print(f"File {i}: {list(epochs.event_id.keys())}")
        raise

    class_names = list(CLASS_TYPE_RANGES.keys())
    class_to_index = {cls: i for i, cls in enumerate(class_names)}

    X_list, Y_list = [], []
    for class_name in class_names:
        idxs = [i for i, label in enumerate(combined_epochs.metadata['class']) if label == class_name]
        if idxs:
            data = combined_epochs[idxs].get_data()
            X_list.append(data)
            Y_list.append(np.full(len(data), class_to_index[class_name]))
        else:
            print(f"Warning: No data found for class '{class_name}'")

    if not X_list:
        raise RuntimeError("No valid class data found after processing.")
    
    X = np.concatenate(X_list, axis=0)  # [N, channels, time_points]
    Y = np.concatenate(Y_list, axis=0)  # [N]

    print(f"Final dataset: {X.shape[0]} samples, {X.shape[1]} channels, {X.shape[2]} time points")
    print(f"Class distribution: {dict(zip(class_names, [np.sum(Y == i) for i in range(len(class_names))]))}")

    return X, Y, class_names, combined_epochs
    
def prepare_data_loaders(X, Y, metadata, args, random_seed=42):
    
    # Select first 64 channels and 500 time points (2s at 250Hz)
    X = X[:, :64, :250]
    
    # Normalize each sample individually
    X = (X - X.mean(axis=(1, 2), keepdims=True)) / (X.std(axis=(1, 2), keepdims=True) + 1e-6)
    
    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.long)
    
    # Apply splits
    X_train, X_temp, Y_train, Y_temp = train_test_split(
        X_tensor, Y_tensor, test_size=0.2, random_state=random_seed, stratify=Y_tensor)
    X_val, X_test, Y_val, Y_test = train_test_split(
        X_temp, Y_temp, test_size=0.5, random_state=random_seed, stratify=Y_temp)
    
    
    # Create DataLoaders
    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)
    test_dataset = TensorDataset(X_test, Y_test)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args['batch_size'], 
        shuffle=True,
        drop_last=True  # Helps with batch normalization
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args['batch_size'], 
        shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args['batch_size'], 
        shuffle=False
    )
    
    return train_loader, val_loader, test_loader


def evaluation(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100. * correct / len(data_loader.dataset)
    avg_loss = total_loss / len(data_loader)
    
    return avg_loss, accuracy

def get_confusion_matrix(model, data_loader, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    cm = confusion_matrix(all_labels, all_preds) 

    class_accs = []
    for i in range(len(class_names)):
        tf = cm[i, i]
        total = np.sum(cm[i, :])
        acc = tf / total if total > 0 else 0
        class_accs.append(acc)
    return cm, class_accs 

def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap ="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def track_best_test_acc(model, test_loader, criterion, device, best_test_acc, best_test_epoch, epoch):
    test_loss, test_acc = evaluation(model, test_loader, criterion, device)

    if test_acc > best_test_acc:
        best_test_acc = test_acc
        best_test_epoch = epoch
    return test_loss, test_acc, best_test_acc, best_test_epoch 

def train(args, model, train_loader, val_loader, test_loader, class_names, save_dir):
    save_dir = create_save_dir(args['save_dir'])
    # Save config
    with open(os.path.join(save_dir, "config.yaml"), "w") as f:
        yaml.dump(args, stream=f, default_flow_style=False, sort_keys=False)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10, factor=0.5)
    
    metrics = Metrics(["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc", 
                       "test_loss", "test_acc", "best_val_acc", "best_test_acc", "best_test_epoch"])
    best_val_acc = 0
    best_test_acc = 0
    best_test_epoch = 0
    
    for epoch in range(args['epochs']):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args['epochs']}"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
        
        train_acc = 100. * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        val_loss, val_acc = evaluation(model, val_loader, criterion, device)
        
        # Test phase (evaluate every epoch)
        #test_loss, test_acc = evaluation(model, test_loader, criterion, device)
        test_loss, test_acc, best_test_acc, best_test_epoch = track_best_test_acc(model, test_loader, criterion, device, best_test_acc, best_test_epoch, epoch+1) 
        # Update learning rate
        scheduler.step(val_acc)
        lr = optimizer.param_groups[0]['lr']
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))
        
        # Print metrics
        print(f"\nEpoch {epoch+1}/{args['epochs']}:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"  Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
        print(f"  Best Val Acc: {best_val_acc:.2f}%")
        print(f" Best Test Acc: {best_test_acc:.2f}% (at epoch {best_test_epoch})") 
        print(f"  Learning Rate: {lr:.6f}")
        
        metrics.add_row([epoch+1, lr, avg_train_loss, train_acc, val_loss, val_acc, 
                         test_loss, test_acc, best_val_acc, best_test_acc, best_test_epoch])
        metrics.save_to_csv(os.path.join(save_dir, "metrics.csv"))
    
    # Final evaluation with best model
    model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pt")))
    test_loss, test_acc = evaluation(model, test_loader, criterion, device)
    print(f"\nFinal Test Results with Best Model:")
    print(f"  Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

    cm, class_accs = get_confusion_matrix(model, test_loader, device, class_names) 
    plot_confusion_matrix(cm, class_names, os.path.join(save_dir, "cm.png"))

    print("\nClass-wise Accuracies:")
    for i, class_name in enumerate(class_names):
        print(f" {class_name}:{class_accs[i]*100:.2f}%")

    highest_idx = np.argmax(class_accs)
    lowest_idx = np.argmin(class_accs)
    print(f"\nHighest Accuracy: {class_names[highest_idx]} ({class_accs[highest_idx]*100:.2f}%)")
    print(f"Lowest Accuracy: {class_names[lowest_idx]} ({class_accs[lowest_idx]*100:.2f}%)")
    
    # Update final test accuracy in metrics
    metrics.df.loc[metrics.df.index[-1], 'test_acc'] = test_acc
    metrics.save_to_csv(os.path.join(save_dir, "metrics.csv"))

    return test_acc, cm, class_accs 

def run_multiple_seeds(args, X, Y, metadata, class_names, num_seeds = 5):
    all_test_accs = []
    all_cms = []
    all_class_accs = []

    main_save_dir = create_save_dir(args['save_dir'])

    for seed in range(num_seeds):
        print(f"\n{'='*50}")
        print(f"Running with random seed {seed+1}/{num_seeds}")
        print(f"{'='*50}")
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        seed_save_dir = os.path.join(main_save_dir, f"seed_{seed+1}")
        os.makedirs(seed_save_dir, exist_ok=True)
        
        train_loader, val_loader, test_loader = prepare_data_loaders(X, Y, metadata, args, random_seed=seed)
        model = NetTraST(args).to(device)

        test_acc, cm, class_accuracies = train(args, model, train_loader, val_loader, test_loader, class_names, seed_save_dir)
        
        all_test_accs.append(test_acc)
        all_cms.append(cm)
        all_class_accs.append(class_accuracies)
    
    avg_test_acc = np.mean(all_test_accs)
    std_test_acc = np.std(all_test_accs)

    total_cm = np.sum(all_cms, axis=0) 
    avg_cm = total_cm.astype(float) / num_seeds
    avg_cm_int = np.round(avg_cm).astype(int) 
    
    avg_class_accs_from_cm = []
    for i in range(len(class_names)):
        true_positives = avg_cm[i, i]
        total_actual = np.sum(avg_cm[i, :])
        class_acc = true_positives / total_actual if total_actual > 0 else 0
        avg_class_accs_from_cm.append(class_acc)
    avg_class_accs_individual = np.mean(all_class_accs, axis=0) 

    print(f"\n{'='*50}")
    print(f"FINAL RESULTS ACROSS {num_seeds} SEEDS")
    print(f"{'='*50}")
    print(f"Average Test Accuracy: {avg_test_acc:.2f}% ± {std_test_acc:.2f}%")
    
    print("\nClass-wise Accuracies (from averaged confusion matrix):")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name}: {avg_class_accs_from_cm[i]*100:.2f}%")
        
    print("\nClass-wise Accuracies (average of individual accuracies):")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name}: {avg_class_accs_individual[i]*100:.2f}%")
    
    # Find highest and lowest accuracy classes on average
    highest_idx = np.argmax(avg_class_accs_from_cm)
    lowest_idx = np.argmin(avg_class_accs_from_cm) 
    print(f"\nHighest Average Accuracy: {class_names[highest_idx]} ({avg_class_accs_from_cm[highest_idx]*100:.2f}%)")
    print(f"Lowest Average Accuracy: {class_names[lowest_idx]} ({avg_class_accs_from_cm[lowest_idx]*100:.2f}%)")
    
    # Plot and save average confusion matrix
    plot_confusion_matrix(avg_cm_int, class_names, os.path.join(main_save_dir, "avg_cm.png"))
    
    
    results_df = pd.DataFrame({
        'Class': class_names,
        'Average Accuracy (CM-based)': [f"{acc*100:.2f}%" for acc in avg_class_accs_from_cm],
        'Average Accuracy (Individual)': [f"{acc*100:.2f}%" for acc in avg_class_accs_individual],
        'Difference': [f"{abs(avg_class_accs_from_cm[i] - avg_class_accs_individual[i])*100:.2f}%" 
                      for i in range(len(class_names))]
    })
    results_df.to_csv(os.path.join(main_save_dir, "final_results.csv"), index=False)
    
   
    with open(os.path.join(main_save_dir, "summary.txt"), "w") as f:
        f.write(f"Average Test Accuracy: {avg_test_acc:.2f}% ± {std_test_acc:.2f}%\n")
        f.write(f"Highest Average Accuracy: {class_names[highest_idx]} ({avg_class_accs_from_cm[highest_idx]*100:.2f}%)\n")
        f.write(f"Lowest Average Accuracy: {class_names[lowest_idx]} ({avg_class_accs_from_cm[lowest_idx]*100:.2f}%)\n")
        f.write(f"\nCalculation Method: Based on averaged confusion matrix\n")
        f.write(f"Number of seeds: {num_seeds}\n")
    return avg_test_acc, avg_cm_int, avg_class_accs_from_cm 


def main():
    # Define your EDF file paths
    subjects = ['s2', 's3', 's4', 's5', 's10', 's12', 's7', 's9', 's11', 's6']  
    edf_paths = [f"{sub}_audio{i} Data.edf" for sub in subjects for i in [1, 2]]
    
    # Filter out missing files (like s10_audio2)
    edf_paths = [f for f in edf_paths if os.path.exists(f)]
    print(f"Found {len(edf_paths)} EDF files for {len(subjects)} subjects")
    
    # Load and preprocess data with enhanced validation
    try:
        X, Y, class_names, combined_epochs = load_and_preprocess_data(edf_paths, session_type='audio')
        metadata = combined_epochs.metadata 
        print(f"Data loaded - X shape: {X.shape}, Y classes: {np.bincount(Y)}")
    except Exception as e:
        print(f"Data loading failed: {str(e)}")
        return
    
    # Model configuration
    args = {
        'class_num': len(class_names),
        'dropout': 0.3,
        'nhead': 4,
        'dim_feedforward': 256,
        'num_layers': 2,
        'embed_dim': 64,  # auto-detect EEG channels
        'vocab_size': 250,  # auto-detect time points
        'kernel_num': 64,
        'kernel_size': 3,
        'batch_size': 64,
        'epochs': 100,
        'early_stopping_patience': 20,
        'lr': 0.0005,
        'save_dir': 'eeg_classification_experiments',
        'device': device
    }

    run_multiple_seeds(args, X, Y, metadata, class_names, num_seeds=5)
    
  
    # # Print model summary
    # print(model)
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"Total parameters: {total_params:,}")
    
    # # Train the model
    # train(args, model, train_loader, val_loader, test_loader)

if __name__ == "__main__":
    main() 
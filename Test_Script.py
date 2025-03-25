# Bailey Hitt - HW 2
# Test Script

import torch                        # For deep learning functionality
import torch.nn as nn               # For neural network modules
import numpy as np                  # For numerical operations
import pandas as pd                 # For handling CSV reading
import matplotlib.pyplot as plt     # For creating plots
from torch.utils.data import TensorDataset, DataLoader, random_split    # For data utilization
from sklearn.preprocessing import StandardScaler                        # For feature normalization
import os                           # For file system operations
import glob                         # For file pattern matching

# Function to build the same network architecture that was used in training
def build_network(input_dim, neuron_count=128, output_classes=4, num_hidden_layers=4):
    # Add first / input layer with ReLU activation
    layers = [nn.Linear(input_dim, neuron_count), nn.ReLU()]
    
    # Add hidden layers 
    for _ in range(num_hidden_layers - 1):      # Loop to create multiple hidden layers
        layers.append(nn.Linear(neuron_count, neuron_count))  # Add a linear layer
        layers.append(nn.ReLU())                # Add ReLU activation after each linear layer
    
    # Add final / output layer with number of classes as output size
    layers.append(nn.Linear(neuron_count, output_classes))
    
    # Create sequential model that combines all layers
    model = nn.Sequential(*layers)
    
    return model

# Load and process data from given CSV files (same as training model)
def load_data(pattern='Vd*_Sg_*.csv'):
    # Find data files matching this pattern
    files = glob.glob(pattern)  #Searches in current directory
    if not files: 
        files = glob.glob(os.path.join('**', pattern), recursive=True)  # Recursive search if files cannot be found

    # Error handling incase files cannot be located
    if not files:
        raise FileNotFoundError(f"No files found matching pattern: {pattern}")
    # Print how many files were found
    print(f"Found {len(files)} data files")
    
    # Initialize data storage
    feature_collection = []     # List to store feature data from all files
    label_collection = []       # List to store label data from all files
    
    # Define columns to extract
    feature_columns = [1, 2, 3]  # Columns for features (2-4)
    target_column = 4  # Column for target/label (5)
    
    # Process each CSV file
    for file_path in files:
        # Read data with pandas
        data_table = pd.read_csv(file_path)
        
        # Extract feature data using column indices
        file_data = data_table.iloc[:, feature_columns].to_numpy()
        
        # Extract target values
        raw_targets = data_table.iloc[:, target_column].to_numpy()
        
        # Process target values with dictionary mapping
        processed_targets = []
        
        # Define mapping for special characters
        target_mapping = {'#': 3}
        
        # Process each target value
        for target in raw_targets:  
            # Check if target is in mapping dictionary
            if target in target_mapping:    #If it's a special character
                processed_targets.append(target_mapping[target])    # Use the mapped value
            else:   # If it's not a special character
                # Convert to integer
                try:    # Try to convert
                    processed_targets.append(int(target))   # Convert to integer and add to list
                except (ValueError, TypeError):             # If conversion fails
                    # Skip invalid values
                    continue        # Skip this value and continue
        
        # Convert to numpy array
        processed_targets = np.array(processed_targets, dtype=np.int64)
        
         # Store data
        feature_collection.append(file_data)         # Add features to the collection
        label_collection.append(processed_targets)   # Add labels to the collection
    
    # Combine all data from all files
    combined_features = np.concatenate(feature_collection, axis=0)
    combined_labels = np.concatenate(label_collection, axis=0)
    
    # Apply data normalization to make training more stable
    normalizer = StandardScaler()    # Create a StandardScaler object
    normalized_features = normalizer.fit_transform(combined_features)   # Normalize the features
    
    # Convert to PyTorch tensors
    features_tensor = torch.tensor(normalized_features, dtype=torch.float32)     # Convert features to tensor
    labels_tensor = torch.tensor(combined_labels, dtype=torch.long)              # Convert labels to tensor
    
    return features_tensor, labels_tensor

# Function to create test data loader
def prepare_test_data(features, labels, batch_size=32):     
    # Create dataset
    dataset = TensorDataset(features, labels)           # Combine features and labels into dataset
    
    # Split ratios: 70% train, 20% validation, 10% test
    total_size = len(dataset)                       # Get total number of samples
    train_size = int(0.7 * total_size)              # Calculate training set size
    val_size = int(0.2 * total_size)                # Calculate validation set size
    test_size = total_size - train_size - val_size  # Calculate test set size
    
    # Split dataset and keep test set
    _, _, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    print(f"Test dataset size: {len(test_dataset)} samples")        #Print test set size
    
    # Create test data loader without shuffling
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return test_loader      # Return test loader

# Function to evaluate the model
def evaluate_model(model, test_loader):
    
    model.eval()    # Set model to evaluation mode
    test_loss = 0.0 # Initialize test loss accumulator
    correct = 0     # Initialize counter for correct predictions
    total = 0       # Initialize counter for total samples
    
    # Lists to store predictions and true labels
    all_predictions = []
    all_true_labels = []
    
    loss_function = nn.CrossEntropyLoss()   # Create loss function for classification
    
    with torch.no_grad():                   # Disable gradient calculation
        for inputs, targets in test_loader: # Loop through batches in test loader
            # Forward pass
            outputs = model(inputs)         # Get model predictions
            loss = loss_function(outputs, targets)  # Calculate loss
            
            # Add batch loss to total loss
            test_loss += loss.item() * inputs.size(0)   
            
            # Get predictions
            _, predicted = torch.max(outputs, 1)
            
            # Count correct predictions
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            # Store predictions and true labels
            all_predictions.extend(predicted.cpu().numpy())
            all_true_labels.extend(targets.cpu().numpy())
    
    # Calculate metrics
    avg_loss = test_loss / total
    accuracy = correct / total
    
    print(f"Test Results - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    # Calculate per-class accuracy
    classes = np.unique(all_true_labels)    # Unique class label
    print("\nClass-wise accuracy:")         # Header
    for cls in classes:
        cls_indices = (np.array(all_true_labels) == cls) # Indicied of samples with this class
        if np.sum(cls_indices) > 0:
            cls_acc = np.sum(np.array(all_predictions)[cls_indices] == cls) / np.sum(cls_indices) # Calculate class accuracy
            print(f"  Class {cls}: {cls_acc:.4f}")      # Print class accuracy
    
    # Create confusion matrix
    cm = np.zeros((len(classes), len(classes)), dtype=int)      # Initialize confusion matrix with zeros
    for i, c1 in enumerate(classes):  # For each true class
        for i, c1 in enumerate(classes):  # For each predicted class
            for j, c2 in enumerate(classes):
                cm[i, j] = sum(1 for a, p in zip(all_true_labels, all_predictions) if a == c1 and p == c2)  # Count occurences
    
    print("\nConfusion Matrix:")    # Header
    print("      ", end="")         # Columb label
    for i in range(len(classes)):   # Row Label
        print(f" Class {classes[i]} ", end="")  # Classes as col headers
    print()
    
    for i, c in enumerate(classes): # For each class as row
        print(f"Class {c}: ", end="")   # Print class label
        for j in range(len(classes)):   # For each column in this row
            print(f"{cm[i, j]:9d}", end="") # Count with formatting
        print()
    
    return accuracy, all_predictions, all_true_labels   # Return eval results

def main():
    
    # Path to saved model
    model_path = 'models/neural_model_acc0.7409.pt'  
    
    # Load data
    features, labels = load_data()
    
    # Prepare test data loader
    test_loader = prepare_test_data(features, labels)
    
    # Create model with same architecture as training
    num_classes = int(torch.max(labels).item()) + 1     # Number of classes
    model = build_network(input_dim=features.shape[1], output_classes=num_classes)  # Create model
    
    # Load saved model weights
    model.load_state_dict(torch.load(model_path))
    
    # Evaluate model
    accuracy, predictions, true_labels = evaluate_model(model, test_loader)
    
    print(f"\nModel evaluation complete. Test accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()

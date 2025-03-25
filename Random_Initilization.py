# Bailey Hitt - HW 2

import torch                 # For deep learning functionality
import torch.nn as nn        # For neural network modules
import torch.optim as optim  # For optimization algorithms 
import numpy as np           # For numerical operations
import pandas as pd          # For handling CSV reading
import matplotlib.pyplot as plt # For creating plots
from torch.utils.data import TensorDataset, DataLoader, random_split # For data utilization
import os                    # For file system operations
import glob                  # For file pattern matching

# Random seed for consistent outputs
torch.manual_seed(1)    # Seed for PyTorch
np.random.seed(1)       # Random seed for NumPy

# Create a neural network with a variable number of hidden layers
def build_network(input_dim, neuron_count=128, output_classes=4, num_hidden_layers=4):
  
    # Add first / input layer
    layers = [nn.Linear(input_dim, neuron_count), nn.ReLU()]
    
    # Add hidden layers 
    for _ in range(num_hidden_layers - 1):
        layers.append(nn.Linear(neuron_count, neuron_count))
        layers.append(nn.ReLU())
    
    # Add final / output layer
    layers.append(nn.Linear(neuron_count, output_classes))
    
    # Create sequential model by unpacking layers list
    model = nn.Sequential(*layers)
    
    # Apply Xavier initialization to all linear layers to help with training stability
    for layer in model:                  # Loop through all layers
        if isinstance(layer, nn.Linear): # Check if linear layer  
            nn.init.xavier_uniform_(layer.weight)   # Apply Xavier initilization to weights
            nn.init.zeros_(layer.bias)   # Initialize biases to 0
    return model    

# Load and process data from given CSV files
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
    
    # Process CSV files
    for file_path in files:
        # Read data with pandas
        data_table = pd.read_csv(file_path) #Load CSV file
        
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
    
    # Convert to PyTorch tensors to be used with PyTorch models
    features_tensor = torch.tensor(combined_features, dtype=torch.float32)
    labels_tensor = torch.tensor(combined_labels, dtype=torch.long)             # Convert labels to tensor
    
    # Print summary stats
    unique_labels, label_counts = np.unique(combined_labels, return_counts=True)    # Get unique labels and counts
    print(f"Processed {features_tensor.shape[0]} total samples with {len(unique_labels)} classes")  # Print sample count
    
    print("\nClass distribution:")   # Print header for class distribution
    for label, count in zip(unique_labels, label_counts):   # For each class
        percentage = (count / len(combined_labels)) * 100   # Calculate percentage
        print(f"  Class {label}: {count} samples ({percentage:.2f}%)")  # Print class stats
    
    return features_tensor, labels_tensor   # Return the processed data

# Split data and create data loaders 
def prepare_data_splits(features, labels):
    # Create dataset by combining features and labels
    dataset = TensorDataset(features, labels)   # Create a TensorDataset object
    
    # Split ratios: 70% train, 20% validation, 10% test
    total_size = len(dataset)   # Get the total number of samples
    train_size = int(0.7 * total_size)  # Calculates training set size
    val_size = int(0.2 * total_size)    # Calculates validation set size
    test_size = total_size - train_size - val_size  # Calculate test set size (remainder)
    
    # Randomly split data up
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
    
    # Print data split information
    print(f"Data split into:")
    print(f"  Training: {train_size} samples ({train_size/total_size:.1%})")
    print(f"  Validation: {val_size} samples ({val_size/total_size:.1%})")
    print(f"  Testing: {test_size} samples ({test_size/total_size:.1%})")
    
    # Create data loaders for batch processing during training
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)   # Create training loader with shuffling
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False)      # Create validation loader w/out shuffling
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False)    # Create test loader w/out shuffling
    
    return train_loader, val_loader, test_loader    # Return all 3

# Train the model and evaluate performance
def execute_training(model, train_loader, val_loader, epochs=500):
    # Setup training parameters
    loss_function = nn.CrossEntropyLoss()   # Create loss function object for classification 
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)   # Create SGD optimizer with 0.001 learning rate
    
    # Initialize tracking metrics lists
    train_loss_history = []     # Track training loss
    val_loss_history = []       # Track validation loss
    train_acc_history = []      # Track training accuracy
    val_acc_history = []        # Track validation accuracy
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()           # Set model to training mode
        epoch_loss, correct_predictions, sample_count=0.0, 0, 0  # Intialize all to 0

        # Process each batch of training data
        for batch_features, batch_labels in train_loader:   # For each batch
            # Reset gradients before each batch
            optimizer.zero_grad()                           
            
            # Forward pass to compute model outputs
            batch_outputs = model(batch_features)
            
            # Calculate loss
            batch_loss = loss_function(batch_outputs, batch_labels)
            
            # Backward pass to compute gradient
            batch_loss.backward()
            
            # Update weights based on gradient
            optimizer.step()
            
            # Get metrics
            epoch_loss += batch_loss.item() * batch_features.size(0)        # Add batch loss to epoch loss
            _, predicted_classes = torch.max(batch_outputs, 1)              # Get predicted classes
            sample_count += batch_labels.size(0)                            # Count samples in batch
            correct_predictions += (predicted_classes == batch_labels).sum().item() # Count correct predictions
        
        # Calculate epoch metrics for training
        avg_train_loss = epoch_loss / sample_count          # Calculate average training loss
        train_accuracy = correct_predictions / sample_count # Calculate training accuracy
        
        # Validation phase
        model.eval()                # Set to evaluation mode
        val_epoch_loss, val_correct, val_count = 0.0, 0, 0  # Initialize all to 0
        
        with torch.no_grad():       # Disable gradient tracking
            for val_features, val_labels in val_loader: # For each validation batch
                # Forward pass only
                val_outputs = model(val_features)       # Get model output
                val_batch_loss = loss_function(val_outputs, val_labels) # Calculate loss
                
                # Get validation metrics
                val_epoch_loss += val_batch_loss.item() * val_features.size(0)  #Add to validation loss
                _, val_predicted = torch.max(val_outputs, 1)                    # Get predicted classes
                val_count += val_labels.size(0)                                 # Count validation samples
                val_correct += (val_predicted == val_labels).sum().item()       # Count correct predictions
        
        # Calculate validation metrics
        avg_val_loss = val_epoch_loss / val_count
        val_accuracy = val_correct / val_count
        
        # Store history for plotting purposed
        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)
        train_acc_history.append(train_accuracy)
        val_acc_history.append(val_accuracy)
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} : "
                  f"Train Loss: {avg_train_loss:.4f}, Acc: {train_accuracy:.4f} - "
                  f"Val Loss: {avg_val_loss:.4f}, Acc: {val_accuracy:.4f}")
    
    print("Training completed!")

    # Return all training history for visualization
    return {
        'train_losses': train_loss_history,     # Return training loss history
        'val_losses': val_loss_history,         # Return validation loss history
        'train_accuracies': train_acc_history,  # Return training accuracy history
        'val_accuracies': val_acc_history       # Return validation accuracy history
    }

def evaluate_model(model, test_loader):
    #Testing model on data and collect predictions
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    # Lists to store predictions and true labels
    all_predictions = []
    all_true_labels = []
    
    loss_function = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            # Forward pass
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            
            # Accumulate loss
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
    
    # Save predictions to file for later analysis if needed later
    # os.makedirs('results', exist_ok=True)
    # np.save('results/predictions.npy', np.array(all_predictions))
    # np.save('results/true_labels.npy', np.array(all_true_labels))
    
    
    # Calculate per-class accuracy
    classes = np.unique(all_true_labels)
    print("\nClass-wise accuracy:")
    for cls in classes:
        cls_indices = (np.array(all_true_labels) == cls)
        if np.sum(cls_indices) > 0:
            cls_acc = np.sum(np.array(all_predictions)[cls_indices] == cls) / np.sum(cls_indices)
            print(f"  Class {cls}: {cls_acc:.4f}")
    
    return accuracy, np.array(all_predictions), np.array(all_true_labels)

# Creating and saving plots
def create_visualizations(history):
    # Create output directory
    os.makedirs('results', exist_ok=True)
    
   # Loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_losses'], label='Training')
    plt.plot(history['val_losses'], label='Validation')
    plt.title('Loss Curves')
    plt.legend()
    plt.grid(alpha=0.5)
    plt.savefig('results/loss_metrics_random_init.png')
    plt.close()  # Close the figure
    
    # Accuracy plot
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_accuracies'], label='Training')
    plt.plot(history['val_accuracies'], label='Validation')
    plt.title('Accuracy Curves')
    plt.legend()
    plt.grid(alpha=0.5)
    plt.savefig('results/accuracy_metrics_random_init.png')
    plt.close()  # Close the figure
    
# Main function to run entire workflow
def run_workflow():
    
    # Load data
    features, labels = load_data()  # Load data from CSV files
    
    # Split data into training, validation and test sets
    train_loader, val_loader, test_loader = prepare_data_splits(features, labels)
    
    # Create neural network model
    num_classes = int(torch.max(labels).item()) + 1 # Get number of classes
    model = build_network(
        input_dim=features.shape[1],# Input features
        neuron_count=128,             # Number of neurons
        output_classes=num_classes, # Number of output classes
        num_hidden_layers=4         # Number of hidden layers
    )
    
    # Train model
    training_history = execute_training(model, train_loader, val_loader)
    
    # Test model and collect predictions
    test_acc, predictions, true_labels = evaluate_model(model, test_loader)
    
    # Create and save visualizations
    create_visualizations(training_history)
    
    # Save trained model
    os.makedirs('models', exist_ok=True)    # Create file directory if it doesnt exist
    model_path = f'models/neural_model_acc{test_acc:.4f}.pt' # Model file name
    torch.save(model.state_dict(), model_path)  # Save model
    print(f"\nModel saved") # Print saved confirmation
    

if __name__ == "__main__":  # Script entry point
    run_workflow()          # Run main workflow function

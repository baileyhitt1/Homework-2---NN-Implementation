# Homework-2---NN Implementation
This code implements a neural network that provides a model for loading data, training a neural network model, and evaluating it's performance.

# Workflow
1. Loads data from CSV files
2. Normalizes features using StandardScaler
3. Splits data into training, validation, and test sets
4. Builds a neural network with configurable architecture 
5. Trains the network using SGD with momentum
6. Evaluates performance on test data
7. Creates visualizations of training metrics
8. Saves the trained model

# Virtual Environment 
I set up and activated a virtual environment on my mac laptop and installed the required packages to be used in my code using the following command in Visual Studio Code:
pip install torch numpy pandas matplotlib scikit-learn

# Functions
- `build_network`: Creates a neural network with variable number of hidden layers and Xavier initialization
- `load_data`: Loads and processes data from CSV files with a specific pattern
- `prepare_data_splits`: Splits data into training, validation, and test sets
- `execute_training`: Trains the model and records performance metrics
- `evaluate_model`: Tests the model on the test set and calculates accuracy
- `create_visualizations`: Creates plots for training/validation loss and accuracy
- `run_workflow`: Main function that orchestrates the entire process
  
# Neural Network Architecture
- Input layer (3 features)
- 4 hidden layers (each with 64 neurons)
- ReLU activation functions between layers
- Output layer (4 classes)
- Xavier initialization
  
# Training Parameters
- Learning rate: 0.001
- Batch size: 32
- Optimizer: SGD
- Loss function: CrossEntropyLoss
- Training epochs: 500
  
# Output
- Training and validation metrics plots
- Test accuracy and per-class performance metrics
- Saved model weights with accuracy in the filename
- This part is commented out but it can save predictions and true labels from the test set for future analysis
  
# Input
The script expects CSV files with ->
- Features in columns 2, 3, and 4
- Class labels in column 5
  

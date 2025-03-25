# Homework-2---NN-Implementation
This document explains all of the changes made to the original code that were made for the comparative analysis of this assignment. 

# With / Without Normalization
In the original code, normalization was used. In order to not have normalization in my code, the changes were simple. I elimated the 2 lines shown below:
 normalizer = StandardScaler()   # Create a StandardScaler object
    normalized_features = normalizer.fit_transform(combined_features)   # Normalize the features
    As well as the associated library. Then I changed "normalized_features" to "combined_features" where the features were being converted to tensor. 

# Different Mini-Batch Size
The alterations for this one were very simple, I just changed the mini batch size from 16 to 128. I chose this large value in order to observe what obvious changes were happening. 

# Different Learning Rate
The modifications for this were also extremely simple, it was just changing the one line of code that initialized the learning rate. The orginial value I chose was 0.001 and I adjusted it to be 0.01, again I chose a large diffence in order to oberve the apparent changes. 

# Changing Optimizers
For this portion of the code, I altered the execute training function to accept the different optimizers that were listed in the homework description. I did this so that the user could simply just change the optimzer type parameter to observe the changes. I added these lines of code to handle each case: 
    # Configure optimizer based on parameter
    if optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'sgd_momentum':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
An error case is also handlded incase the user enters an incorrect optimizer type. 

# Changing Weight Initilization
The orginial code used Xavier Initilization, so in order to change the model to Random Initilization, I just had to change the initilize Xavier - nn.init.xavier_uniform_(layer.weight) to initilize Random - nn.init.normal_(layer.weight, mean=0.0, std=0.01)

# With and Without L2 Regularization 
The original code did not have L2 regularization so to add it into the script, I added a penalty for when the weights get too large. This was done by simply setting weight_decay to a value of my choice (0.9)


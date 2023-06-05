import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split, GridSearchCV
from  sklearn import metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
torch.manual_seed(0)

                        

class Regressor():

    def __init__(self, x, neurons = [64, 64], nb_epoch = 1000, learning_rate=0.1, batch_size = 64, shuffle_flag = True, p = 0.):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """ 
        Initialise the model.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epochs to train the network.
            - neurons {list} -- Number of neurons in each linear layer 
                represented as a list. The length of the list determines the 
                number of linear layers.
            - activations {list} -- List of the activation functions to apply 
                to the output of each linear layer. Apart from the final output layer, 
                it is always set to be a "linear activation function" 
            neurons, activations,

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Replace this code with your own
        X, _ = self._preprocessor(x, training = True)
        self.input_size = X.shape[1]  # number of neurons in the input layers
        self.output_size = 1 # number of neurons in the output layers
        self.nb_epoch = nb_epoch # number of epochs to train
        self.neurons = neurons # architecture of the hidden layer
        self.learning_rate = learning_rate # initial learning rate of the ADAM optimizer
        self.shuffle_flag = shuffle_flag # whether to shuffle or not
        self.batch_size = batch_size # size of the batch to use
        self.training_loss = [] # to store training loss per epoch
        self.p = p # setting the propability of dropout 
        self.x = x # stores the input data
        self.lb = LabelBinarizer() # stores the input data
        self.training_mins = [] # stores the minimal values for the preprocessing from the training data
        self.training_maxs = [] # stores the maximal values for the preprocessing from the training data

        layer_list = []
        n_in = self.input_size
        
        # create hidden layers for given architecture
        for i in neurons:
            p = self.p
            layer_list.append(nn.Linear(n_in,i)) 
            layer_list.append(nn.ReLU())
            layer_list.append(nn.Dropout(p))
            n_in = i
        layer_list.append(nn.Linear(neurons[-1], 1))
        

        # create a model from the hiddel layers
        self.model = nn.Sequential(*layer_list)

        # initiate an optimizer (ADAM optimizer)
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # initiate a loss function
        self.loss = nn.MSELoss()


        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


    def get_params(self, deep = False):
        """ Returns the parameters of the model 

        Function only utilised by GridSearchCV()
        """
        params = {'neurons':self.neurons, 'learning_rate': self.learning_rate, 'nb_epoch': self.nb_epoch, 'p': self.p, 'x': self.x}
        return params

    def set_params(self, **parameters):
        """ Sets the parameters of the models using a parameter dictionary

        Function only utilised by GridSearchCV()
        """
        for param, value in parameters.items():
            setattr(self, param, value)
        return self 

    def get_min_maxs(self, data):
        """ Returns the minimal and maximal values per feature from the training data for normalisation

        """

        if len(data.shape) == 1:
            data = data.reshape(-1,1)
        self.training_mins = [data[:,i].min() for i in range(0,data.shape[1])]
        self.training_maxs = [data[:,i].max() for i in range(0,data.shape[1])]

    def apply_normalisation(self, data):
        """ Normalises the input data using min-max normalisation

        """
        if len(data.shape) == 1:
            data = data.reshape(-1,1)
        for i in range(0,data.shape[1]):
            if(self.training_mins[i] != self.training_maxs[i]):
                data[:,i] = (data[:,i] - self.training_mins[i])/(self.training_maxs[i] - self.training_mins[i])
            else:
                data[:,i] = data[:,i]-self.training_mins[i]

        return data

    def _preprocessor(self, x, y = None, training = False):
        """ 
        Preprocess input of the network.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or 
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size). The input_size does not have to be the same as the input_size for x above.
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).
            
        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        
        # calculate the number of categorical and continious features
        non_num_col_idx = []
        num_col_idx = []
        for i, col_datatypes in enumerate(x.dtypes): 
            if col_datatypes != 'float64' and col_datatypes != 'int64':
                non_num_col_idx.append(i)
            else:
                num_col_idx.append(i)

        # Deal with missing input in the output (y) if y is provided
        if(not y is None):
            for col in y.columns:
                # substitute missing entries with the median value in y
                y = y.fillna(value = {col: np.float64(y[col].median())})
            y_numpy = y.to_numpy()

        

        if len(non_num_col_idx) > 0:
            
            
            for col_idx in non_num_col_idx:
                col = x.columns[col_idx]
                x_column_of_interest = x[col]
                # substitute missing entries with the most frequent category for this feature
                x = x.fillna(value = {col: x_column_of_interest.mode().iloc[0]})
            x_numpy = x.to_numpy()
            x_numpy_categorical = x_numpy[:,non_num_col_idx]
            
            # if we're training the model, ifentify the range of categories for each categorical feature
            if training:
                x_cat_set = []
                for col in range(x_numpy_categorical.shape[1]):
                    x_cat_set.append(set(x_numpy_categorical[:,col]))

            

            x_numpy_cat_pp = np.zeros(x_numpy_categorical.shape[0])
            for col in range(x_numpy_categorical.shape[1]):
                # if we're training the model, save the range of the categories
                if training:
                    lb = LabelBinarizer()  
                    lb.fit(list(x_cat_set[col]))
                    self.lb = lb
                else:
                    lb = self.lb
                one_hot_encoded = lb.transform(x_numpy_categorical[:,col-1])
                if col == 0:
                    x_numpy_cat_pp = one_hot_encoded
                else:
                    x_numpy_cat_pp = np.concatenate((x_numpy_cat_pp, one_hot_encoded), axis = 1)
            

        if len(num_col_idx) > 0:
            for col_idx in num_col_idx:
                col = x.columns[col_idx]
                x_column_of_interest = x[col]
                x = x.fillna(value = {col: x_column_of_interest.median()})
            x_numpy_numerical = x.iloc[:,num_col_idx].to_numpy()

            if training:
                self.get_min_maxs(x_numpy_numerical)
                x_num_pp  = self.apply_normalisation(x_numpy_numerical)
            if not training: 
                x_num_pp  = self.apply_normalisation(x_numpy_numerical)
                

            
        if len(num_col_idx) > 0 and len(non_num_col_idx) > 0:
            x_preprocessed = np.concatenate((x_num_pp, x_numpy_cat_pp), axis = 1)
        elif len(num_col_idx) > 0 and len(non_num_col_idx) == 0:
            x_preprocessed = x_num_pp
        elif len(num_col_idx) == 0 and len(non_num_col_idx) > 0:
            x_preprocessed = x_numpy_cat_pp


        x_tensor = torch.from_numpy(x_preprocessed).float()
        if(not y is None):
            y_tensor = torch.from_numpy(y_numpy).float()

        return x_tensor, (y_tensor if isinstance(y, pd.DataFrame) else None)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


    def batch(self, input_dataset, target_dataset):
        """
        Returns batched version of inputs.

        Arguments:
            - input_dataset -- Array of input features, of shape
                (#_data_points, n_features) or (#_data_points,).
            - target_dataset -- Array of corresponding targets, of
                shape (#_data_points, #output_neurons).

        Returns: 
            - split_batches list[tuple] -- batched data.
            
        """
        # Join target data to inputs.
        dataset = np.c_[input_dataset, target_dataset]
        # Get remainder in the case where batch size doesn't divide number of inputs.
        remainder = dataset.shape[0] % self.batch_size
        # Calculate number of required splits to make in the data.
        num_splits = dataset.shape[0] // self.batch_size
        # Create list of indices to split the data at.
        split_inds = list(range(self.batch_size,self.batch_size*num_splits-1, self.batch_size))
        # If last batch is too small split the data before evenly and append last batch, otherwise split data into batches.
        if(remainder != 0):
            batches = np.split(dataset[:-remainder,:], split_inds, axis=0)
            batches.append(dataset[-remainder:])
        else:
            batches = np.split(dataset, split_inds, axis=0)
        split_batches = []
        # For each batch split back into input data and target data.
        for batch in batches:
            batch_inputs = batch[:,:-target_dataset.shape[1]]
            batch_targets = batch[:,-target_dataset.shape[1]:]
            split_batches.append((batch_inputs,batch_targets))
        return split_batches


    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Apply preporcessing to input data.
        X, y = self._preprocessor(x, y = y, training = True)
        self.model.train()
        dataset = torch.utils.data.TensorDataset(X,y)
        # Create data loader for shuffling and batching the data.
        loader = torch.utils.data.DataLoader(dataset, self.batch_size, shuffle=self.shuffle_flag)
        # Repeat for specified number of epochs.
        for epoch in range(0,self.nb_epoch):
            #Track loss for epoch.
            running_loss = 0.0
            # Loop through batches.
            for batch_num, (X_batch, y_batch) in enumerate(loader):
                # Reset gradients.
                self.optimiser.zero_grad()
                # Perform forward pass.
                outputs = self.model(X_batch)
                # Calculate loss.
                loss = self.loss(outputs, y_batch)
                # Perform backward pass.
                loss.backward()
                # Step model parameters.
                self.optimiser.step()

                running_loss += loss.item()

            self.training_loss.append(np.sqrt(running_loss/len(loader)))
            # if epoch % 1 == 0:
            #     print(f'[{epoch + 1}, {batch_num + 1:5d}] loss: {np.sqrt(running_loss/len(loader))}')

        self.model = self.model

        return self



        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

            
    def predict(self, x):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.ndarray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Apply preprocessing to data.
        X, _ = self._preprocessor(x, training = False) # Do not forget
        # Perform forward pass to get predictions.
        y_pred = self.model(X)
        y_pred = y_pred.detach().numpy()
        
        return(y_pred)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        # Apply preprocessing to data.
        _, Y = self._preprocessor(x, y = y, training = False)
        # Make predictions.
        y_preds = self.predict(x)
        # Calculate RMSE score.
        rmse_score = metrics.mean_squared_error(Y.detach().numpy(),y_preds, squared = False)

        return rmse_score

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def save_regressor(trained_model): 
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor(): 
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model



def RegressorHyperParameterSearch(X, Y, ifmanual = False,  test_size = 0.2, val_size = 0.25): 
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        X - input dataset (to be split into training and validation dataset)
        Add whatever inputs you need.
        
    Returns:
        The function should return your optimised hyper-parameters. 

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    # if the we choose to perform the optimisation using GridSearchCV
    if not ifmanual:
        # Get the test-train split using the inputted train/test ratio
        X_train_init, X_test, Y_train_init, Y_test = train_test_split(X,Y, test_size = test_size, random_state=0)


        # create a list of neuronal architectures to be explored 
        num_neurons_list = [int(2**i) for i in  range(4, 7)] # range of neurons per layer to be explored
        num_layers_list = range(1, 4) # range of layers to be explored
        
        # create architectures with the same number of neurons per layer
        same_number_neurons= [[num_neurons]*num_layers for num_neurons in num_neurons_list  for num_layers in num_layers_list] 

        # create architecture with decreasing number of neurons per layer
        decreasing_number_neurons = [[64, 32, 16], [64, 32], [32, 16]]

        # concatinate into final architecture list
        architecture_list = same_number_neurons + decreasing_number_neurons
        
        # define a list of dropout probabilities to explore
        p_list = [0, 0.2]

        # define a list of learing rates to explore
        learning_rates=[0.005, 0.01, 0.05, 0.1]
        
        # create an input dictionary for GridSearchCV that describes the combination of hyperparameters to explore
        params =  {'neurons': architecture_list, 'learning_rate': learning_rates, 'nb_epoch': [1000], 'p': p_list}

        # create and fit the GridSearch CV class to the training data
        classifier = GridSearchCV(estimator=Regressor(X_train_init), cv=5, param_grid=params, verbose=3,
                                scoring=['neg_root_mean_squared_error', 'r2'], refit='neg_root_mean_squared_error')
        classifier.fit(X_train_init, Y_train_init)


        print(f'Best parameters: {classifier.best_params_}')
        print(f'Best score" {classifier.best_score_}')
        print(classifier.best_estimator_)
        
        # save the best model so fat and print out the results
        save_regressor(classifier.best_estimator_)
        optimal_regressor = classifier.best_estimator_
        optimal_regressor.fit(X_train_init, Y_train_init)
        final_rmse = optimal_regressor.score(X_test, Y_test)
        print(f'Test performance of the best model: {final_rmse}')
        
        
        

        # define the number of epochs to be explored 
        nb_epochs_list = [1000*i for i in range(1, 6)]
        nb_list_start = [100, 250, 500, 750]


        # perform a GridSearch on the most optimal number of eopchs using the most optimal model identified so far
        nb_epochs_list = nb_list_start +  nb_epochs_list 
        params_epochs =  {'neurons': [[64, 64]], 'learning_rate': [0.1], 'nb_epoch': nb_epochs_list, 'p': [0]}
        classifier_epochs = GridSearchCV(estimator=Regressor(X_train_init), cv=5, param_grid= params_epochs, verbose=3,
                                scoring=['neg_root_mean_squared_error', 'r2'], refit='neg_root_mean_squared_error')
        classifier_epochs.fit(X_train_init, Y_train_init)     
        optimal_regressor_epochs = classifier_epochs.best_estimator_
        optimal_regressor_epochs.fit(X_train_init, Y_train_init)
        scores = optimal_regressor_epochs.score(X_test, Y_test)
        print(f'Test performance of the best model: {scores}')    

        # return the RMSE error of the optimal model, the optimal model and the performance of all hyperparameters 
        return scores, optimal_regressor_epochs, classifier_epochs.cv_results_

    
    # perform a grid search manually if ifmanual = True
    else: 
        # split the input data into training, test and validation sets
        X_train_init, X_test, Y_train_init, Y_test = train_test_split(X,Y, test_size = test_size, random_state=0)
        X_train,  X_val, Y_train, Y_val  = train_test_split(X_train_init,Y_train_init, test_size = val_size, random_state=1)

        # define the initial hyperparams to carry out the extensive GridSearch over
        num_neurons_list = [i*10 for i in  range(1, 6)]
        num_layers_list = range(1, 6)
        learning_rates=[0.05, 0.1, 0.2]


        # perform the manual GridSearch
        rmse_params_val = np.zeros ((len(learning_rates), len(num_neurons_list), len(num_layers_list)))
        for i,lr in enumerate(learning_rates):
            for j,num_neurons in enumerate(num_neurons_list):
                for k,num_layers in enumerate(num_layers_list): 
                    neurons = [num_neurons]* num_layers
                    regressor = Regressor(X_train, neurons = neurons, nb_epoch = 4000, learning_rate=lr)
                    regressor.fit(X_train, Y_train)
                    save_regressor(regressor)

                    # Error on validation set
                    rmse_params_val[i,j,k] = regressor.score(X_val, Y_val)
                    print(f'learning rate = {lr}')
                    print(f'number of neurons = {num_neurons}')
                    print(f'number of layers = {num_layers}')
                    print(f'RMSE = {rmse_params_val[i,j,k]}')

        best_perfomance  = np.amin(rmse_params_val)
        print('Best performance is:')
        print(best_perfomance)

        np.save('/Users/anastasia/Desktop/intro2ML/Neural_Networks_100/rmse_params_val',rmse_params_val)
        opt_lr_idx, opt_num_neuron_idx, opt_num_layers_idx = np.unravel_index(np.argmin(rmse_params_val, axis = None), rmse_params_val.shape)
        print('Best learning rate:')
        opt_lr = learning_rates[opt_lr_idx]
        print(learning_rates[opt_lr_idx])

        print('Best neuron number')
        opt_num_neurons = num_neurons_list[opt_num_neuron_idx]
        print(num_neurons_list[opt_num_neuron_idx])

        print('Best layer number')
        opt_num_layers = num_layers_list[opt_num_layers_idx]
        print(num_layers_list[opt_num_layers_idx])


        optimal_architecture = [opt_num_neurons]*opt_num_layers
        # in the best identified model, explore the effects of introducing dropout regularisation of different probabilities
        print('Applying Dropout')

        p_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        rmse_params_val_dropout = np.zeros(len(p_list)+1)
        regressors_list_dropout = []
        rmse_params_val_dropout[0] = best_perfomance
        for p_ind,p_value in enumerate(p_list):
            regressor = Regressor(X_train, neurons = optimal_architecture, nb_epoch = 4000, learning_rate=opt_lr, p = p_value)
            regressor.fit(X_train, Y_train)
            save_regressor(regressor)
            regressors_list_dropout.append(regressor)
            # Error on validation set
            rmse_params_val_dropout[p_ind+1] = regressor.score(X_val, Y_val)
            print(f'learning rate = {opt_lr}')
            print(f'number of neurons = {opt_num_neurons}')
            print(f'number of layers = {opt_num_layers}')
            print(f'Dropout probability = {p_value}')
            print(f'RMSE = {rmse_params_val_dropout[p_ind+1]}')

        # identify the best model so far (minimal error )
        best_perfomance_dropout  = np.amin(rmse_params_val_dropout)
        print(f'Best dropout performance: {best_perfomance_dropout}')

        np.save('/Users/anastasia/Desktop/intro2ML/Neural_Networks_100/rmse_params_val_dropout', rmse_params_val_dropout)
        best_dropout_p_ind = np.argmin(rmse_params_val_dropout)
        best_regressor = regressors_list_dropout[best_dropout_p_ind]
        opt_p = p_list[best_dropout_p_ind]
        scores = best_regressor.score(X_test, Y_test)
        print(f'Test performance of the best model: {final_rmse}')
        
        # select the optimal number of epochs 
        nb_epochs_list = range(500, 10500, 500)
        rmse_params_val_epochs = np.zeros(len(nb_epochs_list))
        for n,nb_epochs in enumerate(nb_epochs_list): 
            regressor = Regressor(X_train, neurons = optimal_architecture, nb_epoch = nb_epochs, learning_rate=opt_lr, p = opt_p)
            regressor.fit(X_train, Y_train)
            save_regressor(regressor)

            # Error on validation set
            rmse_params_val_epochs[n] = regressor.score(X_val, Y_val)
            print(f'number of epochs ={nb_epochs}')
            print(f'RMSE = {rmse_params_val_epochs[n]}')
        
        # identify the best model so far (minimal loss)
        best_perfomance_epochs  = np.amin(rmse_params_val_epochs)
        best_nb_epochs_ind = np.argmin(rmse_params_val_epochs)
        opt_nb_epochs = nb_epochs_list[best_nb_epochs_ind]
        print(f'Best epochs performance: {best_perfomance_epochs} for number of epochs {opt_nb_epochs}')

        np.save('/Users/anastasia/Desktop/intro2ML/Neural_Networks_100/rmse_params_val_epochs', rmse_params_val_epochs)

        # construct a dictionary containing the optimal parameters and the rMSE for each of the combinations explored
        results  = {'architecture': optimal_architecture, 'lr': opt_lr, 'p': opt_p, 'epochs': opt_nb_epochs, 'rmse_params_val': rmse_params_val, 'rmse_params_val_dropout': rmse_params_val_dropout, 'rmse_params_val_epochs': rmse_params_val_epochs}

        # save the most optimal model 
        print(f'Optimal parameters: num of neurons - {opt_num_neurons}, num of layers - {opt_num_layers}, learning rate - {opt_lr}, dropout probability - {opt_p}, number of epochs - {opt_nb_epochs}')
        optimal_regressor = Regressor(X_train, neurons = optimal_architecture, nb_epoch = 4000, learning_rate=opt_lr, p = opt_p)
        optimal_regressor.fit(X_train, Y_train)
        save_regressor(optimal_regressor)
        scores = regressor.score(X_test, Y_test)
        print(f'Test performance of the best model: {scores}')

        # return the final test scores of the optimal values
        return  scores, optimal_regressor_epochs, results
    

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################



def example_main():

    # output_label = "median_house_value"

    # # Use pandas to read CSV data as it contains various object types
    # # Feel free to use another CSV reader tool
    # # But remember that LabTS tests take Pandas DataFrame as inputs
    # data = pd.read_csv("housing.csv") 

    # # Splitting input and output
    # x_train = data.loc[:, data.columns != output_label]
    # y_train = data.loc[:, [output_label]]

    # # Training
    # # This example trains on the whole available dataset. 
    # # You probably want to separate some held-out data 
    # # to make sure the model isn't overfitting
    # regressor = Regressor(x_train, nb_epoch = 1000)
    # regressor.fit(x_train, y_train)
    # save_regressor(regressor)

    # # Error
    # error = regressor.score(x_train, y_train)
    # print("\nRegressor error: {}\n".format(error))

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv") 

    # Splitting input and output
    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]

    # # Training
    # # This example trains on the whole available dataset. 
    # # You probably want to separate some held-out data 
    # # to make sure the model isn't overfitting
    # architectures = [[64,64]]
    # plt.figure(figsize=(12,8))
    # for architecture in architectures:
    #     regressor = Regressor(x_train, neurons = architecture, nb_epoch = 5000, learning_rate=0.1, batch_size = 64, shuffle_flag = True, p = 0.)
    #     regressor.fit(x_train, y_train)
    #     #save_regressor(regressor)

        
    #     x_epoch = np.arange(5000)
    #     plt.plot(x_epoch, regressor.training_loss, label = f"Architecture: {architecture}")
    # plt.xlabel("epochs", fontsize = 22)
    # plt.ylabel("RMSE", fontsize = 22)
    # plt.legend(loc='upper right', fontsize = 22)
    # plt.xticks(fontsize = 22)
    # plt.yticks(fontsize = 22)
    # plt.show()

    architecture = [64,64]
    X_train, X_test, Y_train, Y_test = train_test_split(x_train,y_train, test_size = 0.2, random_state=0)
    regressor = Regressor(X_train, neurons = architecture, nb_epoch = 5000, learning_rate=0.1, batch_size = 64, shuffle_flag = True, p = 0.)
    regressor.fit(X_train, Y_train)
    #error = regressor.score(X_test, Y_test)
    xs = np.arange(50000)
    ys = np.arange(50000)
    y_preds = regressor.predict(X_test)
    plt.figure(figsize=(12,8))
    plt.scatter(Y_test, y_preds)
    plt.plot(xs,ys)
    plt.xlabel("Predicted house median values", fontsize = 22)
    plt.ylabel("True house median values", fontsize = 22)
    plt.xticks(fontsize = 22)
    plt.yticks(fontsize = 22)
    plt.show()



    # Error
    #error, _ = regressor.score(x_train, y_train)
    print("\nRegressor error: {}\n".format(error))
    


if __name__ == "__main__":
    example_main()
    
    


import os
import json
import time
import torch
import random
import pickle
import argparse
import parameters
import numpy as np
import pandas as pd
import prop_pred_model
import matplotlib.pyplot as plt

from tqdm import tqdm
from rdkit import Chem
from rdkit import RDLogger 
from matplotlib import ticker
from collections import Counter
from utils import LoadSmilesData
from sklearn.manifold import TSNE
from skorch import NeuralNetRegressor
from sklearn.decomposition import PCA
from grammar_model import GrammarModel
from prop_pred_model import FeedForward
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
RDLogger.DisableLog('rdApp.*') 

seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Parsing the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--path', help='Path to the weights file', type=str)
parser.add_argument('--plot', help='Plot the latent space', action='store_true')
parser.add_argument('--evaluation', help='Evaluate the model on prior validity, novelty and uniqueness', action='store_true')
parser.add_argument('--train_property', help='Train and evaluate the property prediction model', action='store_true')
parser.add_argument('--hyper_optim', help='Perform hyperparameter optimization over the property prediction model', action='store_true')
parser.add_argument('--reconstruction', help='Estimates the reconstruction accuracy', action='store_true')
parser.add_argument('--active_units', help='Calculates the number of active units', action='store_true'))
args = parser.parse_args()
path = args.path

# Folder to save the evaluations
evaluation_path = os.path.join(path, 'evaluation')
if not os.path.exists(evaluation_path):
    os.makedirs(evaluation_path)

# loading the weights for the encoder and decoder
encoder_weights = os.path.join(path, 'gvae_encoder.pth')
decoder_weights = os.path.join(path, 'gvae_decoder.pth')

params = parameters.load_params()

# Loading non-normalized property information. Normalization can be done later if needed
data = LoadSmilesData(params['labels_path'], normalization='none')

# SMILES/Property for training and testing
smiles_train = data.smiles_train()
smiles_test = data.smiles_test()
property_train = data.property_train()
property_test = data.property_test()

model = GrammarModel(params)

if torch.cuda.is_available():
    model._encoder.cuda()
    model._decoder.cuda()
    
model._encoder.load_state_dict(torch.load(encoder_weights, map_location = device))
model._decoder.load_state_dict(torch.load(decoder_weights, map_location = device))

# If encoded data does not exists on args.path, encode SMILES for training. I'm using standard names for all the callable files
encoded_data_name = os.path.join(evaluation_path, 'encoded_data.pkl')

model._encoder.eval()
model._decoder.eval()
    
if not os.path.exists(encoded_data_name):
    # Encoding the training data
    print('\nEncoding the training data. This might take some time...\n')
    z = model.encode(smiles_train).cpu().detach().numpy()

    with open(encoded_data_name, 'wb') as file:
        pickle.dump(z, file)

else:
    # Loading the encoded data
    with open(encoded_data_name, 'rb') as file:
        z = pickle.load(file)
        
    print('\nThe encoded data already exists and has been loaded!\n')


def normalize_data():
    """
    Normalize the property values according to settings in the parameters.py file
    """
    
    # normalizing the property data
    if params['normalization'].lower().strip() == 'minmax':
        minmaxscaler = MinMaxScaler()
        property_train_normalized = minmaxscaler.fit_transform(np.asarray(property_train).reshape(-1,1))
        scaler = minmaxscaler
        
    elif params['normalization'].lower().strip() == 'standard':
        standardscaler = StandardScaler()
        property_train_normalized = standardscaler.fit_transform(np.asarray(property_train).reshape(-1,1))
        scaler = standardscaler
        
    elif params['normalization'].lower().strip() == 'none':
        property_train_normalized = property_train
        
    return property_train_normalized, scaler
    

def plot_latent_space(t_sne=True):
    """
    Plots the latent space of a trained model. The number of data points to plot and the plot title are set in
    the parameter.py file.
    """
    pca = PCA(n_components=2, random_state=1)
    pca_data = pca.fit_transform(z[:params['n_data_plot']])
    im = plt.scatter(pca_data[:, 0], pca_data[:, 1], c=property_train[:params['n_data_plot']], cmap='viridis', s=5)
    cbar = plt.colorbar()
    tick_locator = ticker.MaxNLocator(nbins=5)
    cbar.locator = tick_locator
    cbar.update_ticks()
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title(params['plot_title'], fontsize=18)
    plt.xlabel('PCA 1', fontsize=17)
    plt.ylabel('PCA 2', fontsize=17)
    plt.locator_params(nbins=5)
    plt.tight_layout()
    plt.savefig(os.path.join(evaluation_path, f"{params['prop']}_latent_space.png"), dpi=300)
    plt.show()
    
    if t_sne:
        tsne = TSNE(n_components=2)
        tsne_data = tsne.fit_transform(z[:params['n_data_plot']])
        im = plt.scatter(tsne_data[:, 0], tsne_data[:, 1], c=property_train[:params['n_data_plot']], cmap='viridis', s=5)
        cbar = plt.colorbar()
        tick_locator = ticker.MaxNLocator(nbins=5)
        cbar.locator = tick_locator
        cbar.update_ticks()
        for t in cbar.ax.get_yticklabels():
            t.set_fontsize(15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.title(params['plot_title'], fontsize=18)
        plt.xlabel('TSNE 1', fontsize=17)
        plt.ylabel('TSNE 2', fontsize=17)
        plt.locator_params(nbins=5)
        plt.tight_layout()
        plt.savefig(os.path.join(evaluation_path, f"{params['prop']}_latent_space_tsne.png"), dpi=300)
        plt.show()
        
        
# evaluating the model on prior validity, percentage of novel molecules and unique molecules --------------------------------
def model_evaluation():
    """
    Evaluates a trained model on prior validity, percentage of novel molecules and percentage of unique molecules
    """
    decodings = []
    valid_decodings_percentage = []
    iterator1 = range(100)
    iterator2 = range(100)

    with open(params['labels_path'], 'rb') as data:
        df = pickle.load(data)
        list_df = list(df.loc[:, 'SMILES_GDB-17'])
        list_df = [Chem.MolToSmiles(Chem.MolFromSmiles(mol), isomericSmiles=True, canonical=True) for mol in list_df]
    
    with tqdm(iterator1, desc='Prior validity') as molecule_set:
      for i in molecule_set:
          aux = []
          z_random = torch.Tensor(np.random.randn(1, 56)).to(device)
          for j in iterator2:
              aux.append(model.decode(z_random)[0])
          
          valid_decodings = [Chem.MolFromSmiles(x) for x in aux]
          valid_decodings[:] = [x for x in valid_decodings if x]
          valid_decodings_percentage.append(len(valid_decodings)/len(iterator2))
          decodings.append(aux)

    decodings = [item for sublist in decodings for item in sublist]
    mols = [Chem.MolFromSmiles(x) for x in decodings]
    mols[:] = [x for x in mols if x]  # Will remove None values from mols
    valid_smiles = [Chem.MolToSmiles(x, isomericSmiles=True, canonical=True) for x in mols]
    valid_smiles[:] = [x for x in valid_smiles if x]  # Will remove None values from valid_smiles
    novel_molecules = [x for x in valid_smiles if x not in list_df]
    
    evaluation = {}

    prior_validity = np.mean(valid_decodings_percentage) * 100
    n_novel_molecules = len(novel_molecules)/len(valid_smiles) * 100
    unique_molecules = len(set(novel_molecules))/len(novel_molecules) * 100
    
    evaluation['prior validity'] = prior_validity
    evaluation['novel molecules'] = n_novel_molecules
    evaluation['unique molecules'] = unique_molecules
    
    print(f'Prior validity: {prior_validity:.2f}%')
    print(f'% of novel molecules from the valid molecules: {n_novel_molecules:.2f}%')
    print(f'% of unique molecules: {unique_molecules:.2f}%')
    
    u_mol = list(set(novel_molecules))
    print(f'\nFrom a total of {len(iterator1) * len(iterator2)} trials, {len(novel_molecules)} new molecules were created.')
    print('Some examples of novel molecules:')
    for smile in u_mol[:20]:
        print(smile)
    
    with open(os.path.join(evaluation_path, 'metrics.json'), 'w') as file:
        json.dump(evaluation, file)
        
        
# evaluating the model on reconstruction accuracy --------------------------------
def reconstruction():
    """
    Test the reconstruction accuracy of the model, i.e., how many time it can correctly
    reconstruct the input molecule
    """
    aux = 0
    mol_avg = []
    with tqdm(smiles_test, desc='Reconstruction accuracy') as molecule_set:
        for molecule in molecule_set:
            for i in range(10):
                z_rec = torch.Tensor(model.encode([molecule])).to(device)
                for j in range(10):
                    x_hat = model.decode(z_rec)[0]
                    if x_hat == molecule:
                        aux += 1
                        
            mol_avg.append(aux/100)
            aux = 0
            
    print(f'The percentage of correct reconstruction is {np.mean(avg) * 100:.2f} %')
    

def active_units(delta=0.01):
    """
    Compute the number of active units
    """
    testloader = DataLoader(smiles_test, batch_size=params['batch'], drop_last=True, shuffle=False)

    for idx, data in enumerate(testloader):
        mu = model.encode(data)
        
        if idx==0:
            batchSum = mu.sum(dim=0, keepdim=True)
            count = mu.size(0)
        else:
            batchSum += mu.sum(dim=0, keepdim=True)
            count += mu.size(0)
            
    testMean = batchSum / count
        
    for idx, data in enumerate(testloader):
        mu = model.encode(data)
        if idx == 0:
            testVarSum = ((mu - testMean) ** 2).sum(dim=0)
        else:
            testVarSum = testVarSum + ((mu - testMean) ** 2).sum(dim=0)
            
    
    testVar = testVarSum / (count - 1)

    # an active unit is given as the number of latent variables that within a test set have a variance higher than delta
    activeUnits = (testVar > delta).sum()
    
    print()
    dim_idx = []
    
    for idx, dim in enumerate(testVar):
        if dim >= delta:
            dim_idx.append(idx)
            print(f'dimension: {idx} - delta: {dim}')
    
    print(f'\nThere are {activeUnits} active units')
    
    return dim_idx
                        
            
def property_model_training():
    """
    Trains and validate the property prediction model using the encoded data as input
    """
    
    def reset_weights(m):
        """
        Reset the models weights for each training trial
        """
        if isinstance(m, torch.nn.Linear):
            m.reset_parameters()
            
            
    # Encoding the testing data 
    print('\nEncoding the testing data...')
    list_z = []
    with torch.no_grad():
        for smile in smiles_test:
            list_z.append(model.encode([smile]).cpu().detach().numpy())
    
    nsamples, batch, ldim = np.shape(list_z)
    list_z = np.asarray(list_z).reshape(nsamples * batch, ldim)
    list_z = torch.Tensor(list_z).to(device)
 
    # preparing the model for training and validations
    property_train_normalized, scaler = normalize_data()
    
    errors = {}
    train_data = []
    testing_error = []
    mae = []
    mse = []
    rmse = []
    
    # associating the property value
    property_train_normalized = property_train_normalized.flatten().tolist()
    
    for i in range(len(z)):
        train_data.append([z[i], property_train_normalized[i]])
    
    chunk = int(params['valid_split'] * len(train_data))  
    train_split, validation_plit = random_split(train_data, [len(train_data) - chunk, chunk])
    
    trainloader = DataLoader(train_split, batch_size=params['batch'], drop_last=True, shuffle=False)
    validloader = DataLoader(validation_plit, batch_size=params['batch'], drop_last=True, shuffle=False)
    
    # loading the model
    pp_model = FeedForward(params['latent_dim'], params['hidden_layer_prop_pred'])
    
    if torch.cuda.is_available():
        pp_model.cuda()
    
    optimizer = torch.optim.Adam(pp_model.parameters(), lr=params['learning_rate_prop'], amsgrad=True)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, min_lr=1e-6, verbose=True)
    criterion = torch.nn.L1Loss()
    
    epochs = 50
    min_valid_loss = np.inf
    
    # will be running the training and testing procedure for 5 times
    for i in range(5):
        # to reset the weights for each trial training
        pp_model.apply(reset_weights)
        valid_loss = []
        
        print(f'\n\033[1m---------- Trial training: {i+1} ----------\033[0m')
        for epoch in range(epochs):

            pp_model.train()
            avg_loss = 0
            for x, label in trainloader:
                
                predictions = pp_model(x.to(device))
                loss = criterion(predictions.view(-1).to(device), label.float().to(device))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                avg_loss += loss.item()

            # validation procedure -----------------------------------------------------
            pp_model.eval()
            
            avg_loss_val = 0
            
            with torch.no_grad():
                for x_val, label_val in validloader:
                    predictions_val = pp_model(x_val.to(device))
                    loss_val = criterion(predictions_val.view(-1).to(device), label_val.float().to(device))
                    
                    avg_loss_val += loss_val.item()
            
                print(f"epoch: {epoch+1}/{epochs}\nmae: {avg_loss/len(trainloader):>5f} ----- mae_val: {avg_loss_val/len(validloader):>5f}") 
                valid_loss.append(avg_loss_val/len(validloader))
                
        if np.mean(valid_loss) < min_valid_loss:
            
            print(f'Validation loss decreased from {min_valid_loss:.6f} to {np.mean(valid_loss):>6f}. Saving the model!')
            torch.save(pp_model.state_dict(), os.path.join(evaluation_path, 'prop_pred_model.pth'))
            
            min_valid_loss = np.mean(valid_loss)
            
        else:
            print(f'\nThe loss didn\'t decrease!')
        
        # after each training and validation, the model will be tested on the hold-out set
        prop_pred_normalized = pp_model(list_z).cpu().detach().numpy()
        prop_pred = scaler.inverse_transform(prop_pred_normalized).flatten().tolist()
        
        mae_loss = mean_absolute_error(property_test, prop_pred)
        mse_loss = mean_squared_error(property_test, prop_pred)
        rmse_loss = mean_squared_error(property_test, prop_pred, squared=False)
        
        mae.append(mae_loss)
        mse.append(mse_loss)
        rmse.append(rmse_loss)
        
    errors['mae'] = np.mean(mae)
    errors['std_mae'] = np.std(mae)
    errors['mse'] = np.mean(mse)
    errors['std_mse'] = np.std(mse)
    errors['rmse'] = np.mean(rmse)
    errors['std_rmse'] = np.std(rmse)
    
    print(f'\nThe MAE of the hold-out set is: {np.mean(mae):.5f} \u00B1 {np.std(mae):.5f}')
    print(f'The MSE of the hold-out set is: {np.mean(mse):.5f} \u00B1 {np.std(mse):.5f}')
    print(f'The RMSE of the hold-out set is: {np.mean(rmse):.5f} \u00B1 {np.std(rmse):.5f}')
    
    with open(os.path.join(evaluation_path, 'prop_metrics.json'), 'w') as file:
            json.dump(errors, file)
            

def hyperparameter_optimization(fraction=0.5):
    """
    Performs a grid search over a number of parameters to improve the property prediction performance
    """
    
    class FeedForwardHyperparameterOptimization(torch.nn.Module):
        """
        This network is the same as used for property prediction. It is defined here for convinience
        """
        def __init__(self, input_dim=56, hidden_units=512, dropout=0.5):
            super(FeedForwardHyperparameterOptimization, self).__init__()
            
            self.feedforward = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_units),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(hidden_units, hidden_units),
                torch.nn.LeakyReLU(),
                torch.nn.Dropout(p=dropout),
                torch.nn.Linear(hidden_units, hidden_units),
                torch.nn.LeakyReLU(),
                torch.nn.Dropout(p=dropout),
                torch.nn.Linear(hidden_units, 1)
                )
        
        def forward(self, x):
            output = self.feedforward(x)
            return output

    pp_model = FeedForwardHyperparameterOptimization()
    
    property_train_normalized, _ = normalize_data()
    x, y = z[:int(len(z) * fraction)], property_train_normalized[:int(len(z) * fraction)]
    
    net = NeuralNetRegressor(pp_model, 
            max_epochs=20,
            lr=0.001,
            optimizer=torch.optim.Adam,
            iterator_train__shuffle=True,
            )
    
    net.set_params(train_split=False, verbose=0)
    hyparameters = {
        'lr': [0.001, 0.0001],
        'max_epochs': [30, 50],
        'module__input_dim': [params['latent_dim']],
        'module__hidden_units': [128, 256, 512],
        'module__dropout': [0.1, 0.3]
    }
    
    gs = GridSearchCV(net, hyparameters, refit=False, cv=3, scoring='neg_mean_squared_error', verbose=2)
    gs.fit(torch.Tensor(x), torch.Tensor(y))
    print(f'Best score: {gs.best_score_:.3f}, best params: {gs.best_params_}')
    
    best_hyper = {}
    best_hyper['lr'] = gs.best_score_['lr']
    best_hyper['max_epochs'] = gs.best_score_['max_epochs']
    best_hyper['module__dropout'] = gs.best_score_['module__dropout']
    best_hyper['module__hidden_units'] = gs.best_score_['module__hidden_units']
    
    with open(os.path.join(evaluation_path, 'prop_best_hyper.json'), 'w') as file:
            json.dump(best_hyper, file)
            
 
        
if __name__ == "__main__":
    
    if args.plot:
        plot_latent_space(t_sne=True)
        
    if args.evaluation:
        model_evaluation()
        
    if args.train_property:
        property_model_training()
        
    if args.hyper_optim:
        hyperparameter_optimization()
        
    if args.reconstruction:
        reconstruction()
        
    if args.active_units:
        active_units()
 
        
        
        

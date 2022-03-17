import numpy as np

"""
Available properties:

'dipole_moment', 'isotropic_polarizability', 'energy_of_HOMO', 'energy_of_LUMO', 'Gap',
'electronic_spatial_extent', 'zero_point_vibrational_energy',
'internal_energy_at_0K', 'internal_energy_at_298.15K',
'enthalpy_at_298.15K', 'free_energy_at_298.15K',
'heat_capacity_at_298.15K'
"""

def load_params():
    """
    Load the parameters and hyperpameters which will be used in the model
    """
    parameters = {
        
        # general
        'epochs': 50,
        'batch': 256,
        'valid_split': 0.1,		# Percentage of data to use as validation
        'dataset_path':'data/qm9_grammar_dataset.h5',
        'labels_path': 'data/QM9_STAR.pkl',
        'prop_div': True,             # Whether or not divide the property value by the number of atoms
        'plot_title': 'LUMO (eV)',
        'n_data_plot': 30000,		# Number of data points for visualization of the latent space
        
        # vae general
        'latent_dim': 56,
        'max_length': 100,
        'anneal_kl': False,
        'n_cycle': 1,                    # num of cycles for KL annealing
        'ratio_anneal_kl': 0.8,
        
        # neural networks
        'n_layers': 3,                   # num of layers for GRU decoder
        'hidden_layer_prop': 70,         # num of neurons of the integrated property network
        'hidden_layer_prop_pred': 256,   # num of neurons of the separated property network
        'learning_rate': 1e-3,
        'learning_rate_prop': 1e-3,
        'prop_weight': 1,                   
        'reconstruction_weight': 1,
        'kl_weight': 1,
        
        # property to be used in training
        'prop_pred': True,               # whether or not use property information for training
        'prop': 'energy_of_LUMO',        
        'normalization': 'standard'        # how to normalize the property information (minmax, standard, none)
    }
    
    return parameters
    

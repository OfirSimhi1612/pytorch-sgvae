import os
import yaml
from typing import Any, Dict
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

"""
Available properties:
TODO: update the parameters
'energy_of_HOMO', 'energy_of_LUMO', 'Gap',
'dipole_moment', 'isotropic_polarizability',
'electronic_spatial_extent', 'zero_point_vibrational_energy',
'internal_energy_at_0K', 'internal_energy_at_298.15K',
'enthalpy_at_298.15K', 'free_energy_at_298.15K',
'heat_capacity_at_298.15K'
"""

def _validate_type(name: str, value: Any, expected_type: type):
    if not isinstance(value, expected_type):
        raise TypeError(f"Parameter '{name}' must be of type {expected_type.__name__}, got {type(value).__name__}")

def _validate_params(params: Dict[str, Any]) -> Dict[str, Any]:
    # General
    _validate_type('epochs', params['epochs'], int)
    _validate_type('batch', params['batch'], int)
    _validate_type('valid_split', params['valid_split'], float)
    if not (0 < params['valid_split'] < 1):
        raise ValueError("'valid_split' must be between 0 and 1")
    _validate_type('dataset_path', params['dataset_path'], str)
    _validate_type('labels_path', params['labels_path'], str)
    _validate_type('prop_div', params['prop_div'], bool)
    _validate_type('plot_title', params['plot_title'], str)
    _validate_type('n_data_plot', params['n_data_plot'], int)

    # VAE general
    _validate_type('latent_dim', params['latent_dim'], int)
    _validate_type('max_length', params['max_length'], int)
    _validate_type('anneal_kl', params['anneal_kl'], bool)
    _validate_type('n_cycle', params['n_cycle'], int)
    _validate_type('ratio_anneal_kl', params['ratio_anneal_kl'], float)

    # Neural networks
    _validate_type('n_layers', params['n_layers'], int)
    _validate_type('hidden_layer_prop', params['hidden_layer_prop'], int)
    _validate_type('hidden_layer_prop_pred', params['hidden_layer_prop_pred'], int)
    _validate_type('learning_rate', params['learning_rate'], float)
    _validate_type('learning_rate_prop', params['learning_rate_prop'], float)
    _validate_type('prop_weight', params['prop_weight'], int)
    _validate_type('reconstruction_weight', params['reconstruction_weight'], int)
    _validate_type('kl_weight', params['kl_weight'], int)

    # Property to be used in training
    _validate_type('prop_pred', params['prop_pred'], bool)
    _validate_type('prop', params['prop'], str)
    if params['normalization'] not in {'minmax', 'standard', 'none'}:
        raise ValueError("'normalization' must be one of: minmax, standard, none")
    return params

def load_params() -> Dict[str, Any]:
    """
    Load the parameters and hyperparameters from the YAML config file, with validation and typing.
    """
    config_path = os.path.join(os.path.dirname(__file__), 'config.yml')
    with open(config_path, 'r') as f:
        parameters = yaml.safe_load(f)
    return _validate_params(parameters)

hyper_params = load_params()

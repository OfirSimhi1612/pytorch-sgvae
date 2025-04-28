import torch

from src.config.hyper_parameters import device
from src.explore.config.exploration_parameters import explore_params
from src.models.prop_pred import PropertyPredictionModel
from src.models.SGVAE import SGVAE


def load_models(path):
    """
    Load the SGVAE model and the property prediction model from the given path.

    Args:
        path (str): Path to the directory containing the model weights.

    Returns:
        tuple: Loaded SGVAE model and property prediction model.
    """
    # Load SGVAE model
    sgvae_model_path = f"{path}/gvae_model.pth"
    sgvae_model = SGVAE()
    sgvae_model.load_state_dict(torch.load(sgvae_model_path, map_location=device))
    sgvae_model.eval()

    # Load property prediction model
    prop_pred_model_path = f"{path}/evaluation/prop_pred_model.pth"
    prop_pred_model = PropertyPredictionModel()
    prop_pred_model.load_state_dict(
        torch.load(prop_pred_model_path, map_location=device)
    )
    prop_pred_model.eval()

    return sgvae_model, prop_pred_model


def main():
    sgvae_model, prop_pred_model = load_models(explore_params["model_path"])


if __name__ == "__main__":
    main()
